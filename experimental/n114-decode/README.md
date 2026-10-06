# N114: fused decode kernels for Qwen3.8-Flash-Next on the B70 (Strata ports, MIT)

Goal: cut the non-MoE decode time (~25 ms/step today) by porting Strata's fused decode kernels
(github.com/Niko1221/Strata, HEAD 82f46a8, MIT; attribution blocks in every `csrc/*.sycl`) to SYCL torch XPU ops,
wired into SGLang 0.5.20 (image 24c872759256) by env-guarded monkeypatches. All three are default off; with the
switch off nothing is patched, and every unsupported case inside a switched-on path calls the original callables.

| target | switch | library / ops | replaces (per decode step) |
|---|---|---|---|
| GDN decode (36 layers) | `EXL3_GDN_DEC_SYCL=1` | `build/gdn_dec.so`, `torch.ops.n114gdn.{conv,rec_norm}` | b/a copies + cat + Triton conv + Triton packed decode + z copy + Triton gated norm (6-7 kernels/layer) -> 2 kernels |
| hyper-connection read (96 halves) | `EXL3_HC_DEC_SYCL=1` (+ `EXL3_HC_DEC_FOLD=1`) | `build/hc_dec.so`, `torch.ops.n114hc.{mix,combine_mix,combine}` | norm + 2 GEMMs + epilogues + combine GEMM + combine (n107-hc: ~7 kernels/half) -> 3-4 kernels, weights read once for up to 8 rows |
| decode QSA (12 layers) | `EXL3_QSA_DEC_SYCL=1` | `build/qsa_dec.so`, `torch.ops.n114qsa.*` | torch expansion + logical->physical + torch reference attention (~40 kernels/layer) |

All kernels: no host syncs, no host reads of device data, outputs from `at::empty` on the current XPU stream:
capturable in the XPU decode graph.

## Files

| file | what |
|---|---|
| `csrc/gdn_dec.sycl` | GDN conv + recurrence/gated-norm kernels (Strata fused_gdn_conv_l2 / fused_gdn_ab / fused_gdn_step_norm) |
| `csrc/hc_dec.sycl` | HC pre (folded combine + norm), down (split-K, Wd + Wi), up (+ silu/sigmoid epilogue), combine (Strata fused_gr_down / fused_gr_up / fused_gr_read_multi) |
| `csrc/qsa_dec.sycl` | decode QSA (Strata qsa_select / qsa_decode_attn) |
| `csrc/cpu_test_{gdn,hc,qsa}.cpp` | the kernels on the OpenCL CPU device (or a host emulator) vs references |
| `csrc/build_sycl.sh`, `csrc/cpu_test.sh` | build / CPU-test inside the image (no GPU) |
| `{gdn,hc,qsa}_dec.py` | loaders, `why_not()`, wrappers |
| `patch_{gdn,hc,qsa}_dec.py` | env-guarded SGLang patches |
| `test_{gdn,hc,qsa}_dec_xpu.py` | B70 tests: correctness vs the current path, XPU graph capture/replay, per-call timing |
| `offline_gdn_sem.py`, `emulate_hc_check.py` | offline (Mac) semantics checks |
| `run_xpu.sh`, `n114_job.sh` | GPU runner (image, 84:00.0 only) and the guarded job wrapper |

## GDN decode (`EXL3_GDN_DEC_SYCL=1`)

On XPU, `num_v_heads / num_k_heads = 3` is not in SGLang's fused-split ratios, so the decode layer runs
`fix_query_key_value_ordering` (q | k | v | z column split of in_proj_qkvz, b | a of in_proj_ba), two `.contiguous()`
copies, `torch.cat(q, k, v)`, Triton `causal_conv1d_update`, Triton `fused_recurrent_gated_delta_rule_packed_decode`
(l2norm, gates, recurrence), a z copy (bs > 1) and the Triton `RMSNormGated`.

n114:
* `conv`: one work-item per (row, channel), reads q|k|v straight from the in_proj_qkvz output (its first 10240
  columns ARE the concatenation), width-4 causal conv with the bf16-rounded products of Triton's bf16 x bf16 multiply
  (`EXL3_GDN_DEC_ROUND_PROD`, default 1), conv-state shift in SGLang's `(slots, C, 3)` view with any strides, silu,
  bf16 out. Padded rows (slot < 0) untouched, like Triton.
* `rec_norm`: one work-group per (row, value head), 128 columns x RG row groups (default RG 4 = 512 work-items,
  sub-group 16). A thread holds 32 consecutive k entries of its column's state row (SGLang's `[slots, H, V, K]` layout);
  the k reductions are sub-group shuffles. q/k l2 norms, `g = -exp(A_log) softplus(a + dt_bias)`,
  `beta = bf16(sigmoid(b))`, decay-then-delta-rule exactly in the packed-decode order, `o` rounded to bf16 where SGLang
  stores it, then the gated RMS norm over the head (work-group reduction) and the bf16 out_proj input.
  Optional `EXL3_GDN_DEC_BA=1`: the in_proj_ba projection (in_proj_a/b are plain F16->bf16 weights in the checkpoint)
  runs inside the kernel (2 x 2560-long dots per head), removing that GEMM.

Wiring (no source edits): the patched `Qwen3_5GatedDeltaNet.forward` (decode only) passes the TRUE q|k|v, a, b as
strided views through the normal `self.attn(...)` call and leaves a thread-local context; the patched
`GDNAttnBackend.forward_decode` reads the layer's conv/ssm cache and slot indices like the original, runs the two
kernels, calls the original `_track_mamba_state_decode`, and the model forward goes straight to `out_proj`. Anything
the fused path does not cover (replay-SSM ring, non-Triton decode kernel, intel_xpu backend, other head ratios,
layouts/dtypes, a kernel error) calls the original `forward_decode` with contiguous copies (= the original inputs) and
the original norm tail. `install()` refuses when the image's sources lack the anchor lines it relies on.
`EXL3_GDN_DEC_CHECK=N` compares the first N eager calls against SGLang's own kernels on compact state copies.

Offline validation (Mac):
* `offline_gdn_sem.py` runs SGLang's own Triton kernels (Triton 3.7.1 interpreter) on SGLang-ordered inputs vs the
  n114 math on the raw projections: conv out, conv state and y bit-identical at bs 1/2/4 (sigmoid and swish gates),
  ssm state within 1.2e-7, once the interpreter's truncating bf16 casts are emulated (`trunc=1`). This validates the
  layouts (q|k|v|z, b|a, head map, z offset, pad rows); the RNE / product-rounding question is settled on the GPU.
* `csrc/cpu_test_gdn.cpp` on a host SYCL emulator: 9 cases (bs 1-8, pad rows, int64 indices, RG 2/4/8, bf16 state,
  silu gate, recurrence-only, ba-fused) ALL PASS: conv out and conv state bit-exact, y 99.96-100 % bf16-equal to the
  reference, fp32 state rel error <= 1.2e-7.

## HC decode (`EXL3_HC_DEC_SYCL=1`, optional `EXL3_HC_DEC_FOLD=1`)

`GatedResidual.mix` / `combine` for XPU bf16 rows 1..8 (`EXL3_HC_DEC_MAX_M`), per-branch norm layout. Three launches
per mix, weights (13.2 MB per half) read once for all rows:
* `pre`: one work-group per (row, branch); optional folded combine of the previous half (`R' = bf16(R + bo * g)`,
  written as the new residual), then the Gemma per-branch norm (`1 + w`) -> bf16 `n`.
* `down`: 21 row blocks x KS splits (KS 8 for M <= 2, else 4), `n` slice staged in local memory, one sub-group per
  Wd / Wi row, fp32 split-K partials (fixed-order sum, deterministic).
* `up`: 80 work-groups; sums the partials, `a = bf16(silu(bf16(t) / hc))`, Wu rows, `u` rounded to bf16, sigmoid * n,
  branch mean -> `mixed`; work-group 0 writes `l = bf16(n . Wi)` for this half's combine.
* `combine`: elementwise, for the non-folded case.
`mix` returns an `HCRes` (tuple subclass carrying `l`); `combine` takes the SYCL path only for an `HCRes` of the same
module. `EXL3_HC_DEC_FOLD=1` folds attn-combine into mlp-mix inside `_prepare_qwen4_exp_mlp` (3 source anchors
checked). Launch config `EXL3_HC_DEC_CFG=ks,uc,mode` (default `0,32,0`). Wd/Wu/Wi must be bf16 (checkpoint F16 is
converted on load); the norm weight may be fp32/bf16/fp16. Offline: `emulate_hc_check.py` (torch CPU inductor)
matches compiled `_mix_compute` 100 / 100 / 99.99 % at M = 1/2/4 and `_combine_compute` 100 %.

## Decode QSA (`EXL3_QSA_DEC_SYCL=1`)

`torch.ops.n114qsa.decode_attn`: chunk kernel (one 256-thread work-group per (row, kv head, split); each cell
reproduces `torch_expand_qsa_block_indices` (block * 4 + offset masked to seq_len, plus the 0-3 pending-tail tokens)
and the logical->physical map (exl3xpu's graph-mode `req_to_token` route or SGLang's `token_slot_table`), fp32
scores / online softmax, fp8 e4m3 / bf16 KV read in place) + log-sum-exp merge kernel. Graph-safe: per-step values
read from device tensors, launch shape fixed per (rows, index width). A second mode takes already-expanded token
indices (verify / draft rows). Selection is NOT ported: exl3xpu's `qsa_decode_select` (one SYCL kernel) stays.
`patch_qsa_dec.py` wraps whatever `_forward_paged_attention` is installed (so install it AFTER exl3xpu's
`qsa_xpu.install()`), defers the block expansion on plain decode batches (no spec info, so MTP never sees block
ids), and falls back to the previous path (re-expanding first) for anything else. Config `EXL3_QSA_DEC_CFG`
(chunk, cpw, sg, fp8, target work-groups; default 64,0,16,1,160), `EXL3_QSA_DEC_MAX_ROWS` (64),
`EXL3_QSA_DEC_CHECK=N`. Offline: `offline_qsa_sem.py` (identical physical slot sets vs today's path, max_abs vs
fp64 7.6e-3 vs today's 1.19e-2) and `mock_patch_qsa_dec.py` (routing, fallbacks bit-identical) pass.

## Status (2026-10-06): built + CPU-validated, NOT run on any GPU

GPU work was put on hold (research focus moved to GLM-5.3-Flash on the 3090); `runs/N107/CLEARED` never appeared
while this was being done, so no B70 job was run.

Done:
* All three libraries build with icpx in image 24c872759256 (`csrc/build_sycl.sh`: gdn 38 s, hc 36 s, qsa 43 s;
  `build/*.so` on omarchy).
* OpenCL CPU-device tests in the image (`csrc/cpu_test.sh`, log `results/cpu_test.log` on omarchy):
  * gdn 9/9 PASS: conv out + conv state bit-exact, y 99.97-100 % bf16-equal to the reference, fp32 state rel
    <= 1.2e-7 (bs 1-8, pad rows, int64 indices, RG 2/4/8, bf16 state, silu gate, rec-only, ba-fused);
  * qsa 50/50 PASS (R 1/2/4/8, fp8 / fp8-via-half / bf16 KV, req_to_token and slot_table routes, expanded and
    logical modes, edge rows; max_abs <= 1.6e-2, all within 0.85 of the tolerance);
  * hc 13/14 at the time of the run: one case had a 2-ulp mix difference in 0.04 % of elements (OpenCL-CPU exp vs
    std::exp); the test tolerance was relaxed to 2 ulp afterwards (not re-run).
* `install()` of all three patches succeeds against the image's SGLang (no GPU; `results/n114_install_smoke.py`):
  source anchors present, GDN `_track_mamba_state_decode(..., layer_id)` signature matches.
* GDN semantics vs SGLang's own Triton kernels (interpreter, `offline_gdn_sem.py`): bit-identical once the
  interpreter's truncating bf16 casts are emulated.

Remaining (in order):
1. GPU correctness + timing on 84:00.0 via the guarded wrapper (checks CLEARED, health, `journalctl -k -b`, D-state):
   ```
   cd ~/freetoken-exl3; Q=kernels/xpu_bmg/n114-decode
   $Q/n114_job.sh gdn 600 $Q/run_xpu.sh test_gdn_dec_xpu.py      # add --quick for a short run
   $Q/n114_job.sh hc 600 $Q/run_xpu.sh test_hc_dec_xpu.py        # --sweep for ks/uc/mode
   $Q/n114_job.sh qsa 600 $Q/run_xpu.sh test_qsa_dec_xpu.py
   ```
   Each compares against the image's current path at bs 1/2/4(/8), captures and replays an XPU graph, prints
   us/layer (eager and graph) for current vs n114, writes `results/<name>_<time>.json`, exits 0 only if all pass.
   Open questions they settle: whether the XPU Triton conv rounds bf16 x bf16 products (`EXL3_GDN_DEC_ROUND_PROD`),
   whether icpx fast-math breaks the HC combine bit-exactness (rebuild with `N114_FLAGS=-fp-model=precise`), and
   the launch configs (GDN rg/grf, HC ks/uc/mode, QSA chunk/cpw/sg).
2. Serve wiring (not done; `runs/N104-nvtier/` scripts are only on omarchy): mount this dir at `/n114`, pass
   `EXL3_{GDN,HC,QSA}_DEC_*`, and in `sglang_plugin.activate()` after exl3xpu's install:
   `sys.path.insert(0, "/n114"); import patch_gdn_dec, patch_hc_dec, patch_qsa_dec; <each>.install()`.
   Bring-up with `EXL3_GDN_DEC_CHECK=4` / `EXL3_QSA_DEC_CHECK=4`, then the reference panel, then the decode A/B.
3. Not ported: Strata's decode block selection (exl3xpu's kernel kept), cross-layer HC fold (mlp combine -> next
   layer's attn mix; PLE layers in between need care), multi-token verify for GDN.

Expected gain (estimates, unmeasured): GDN ~150-180 fewer graph nodes/step (+ 36 if `EXL3_GDN_DEC_BA=1`), HC ~380
fewer nodes/step with weights read once at bs 2/4 (HC weight streaming is ~1.26 GB/step, ~2.6 ms floor), QSA ~45 -> 2
nodes per layer (~500 fewer/step). At ~5 us per graph node that is roughly 5 ms of the ~25 ms non-MoE decode step,
plus kernel-time gains the GPU tests have to show.

# 19. What Strata (github.com/Niko1221/Strata, MIT) teaches us for Qwen3.8-Flash-Next on the B70 (2026-10-06)

Source: two read-only analysis agents over Strata HEAD 82f46a8 (docs/INTEL.md, docs/INTEL_ARC.md, sycl/, src/, paper).
Clone: Mac scratchpad strata-src/Strata (untrusted data, never built).

## Headline
- Strata has a SYCL port measured on an Arc Pro B70 32 GB (PCIe 3.0 x8 host). Same model family (512 experts top-10).
  - Coder IQ1_M (256 experts, all in VRAM): decode 78 tok/s (MTP), prefill 790 @2K / 986 @8K / ~1,160 @40K.
  - Full 512-expert IQ2_XS (18,329 experts in VRAM + 6,247 in a USM-host mirror): decode 58.6-64 tok/s (MTP), prefill 549 @2.2K.
- Their raw B70 forward is NOT faster than ours: one verify round (up to 6 tokens, 48 layers) ~38 ms.
  Their decode lead = MTP self-speculation (2.6-3.4 tokens/round; 77-87% accept on code, 55-72% prose)
  + no NVMe in decode (missed experts read by the GPU from a USM-host mirror, zero-copy; no SVM).
- Their 8K prefill on B70 (Coder): MoE ~3.8 s, attention ~1.1 s, rest ~2.8 s. Ours: MoE 1.2 s, QSA 3.1 s, rest 5-7 s.
  Our MoE path is ~3x better; we lose on QSA and the non-MoE remainder.

## Ranked plan for us (estimates; re-measure on our Gen4 x16 B70)
1. QSA prompt pipeline port (format-independent): indexer append batch, warp block scores, register radix top-k
   (exact selection rule), per-query split-K FP32 attention reading KV in place via the page table, no union.
   Files: sycl/src/kernels/cuda/{native_qsa_indexer,qsa_select,qsa_decode_attn}.dp.cpp, orchestration sycl/src/prefill/prefill.cpp:2270-2520.
   They measured the union idea: union of 8 neighbours' selections = 3.3x one query's (16: 5.1x), only 12% shared.
   Expected: QSA 3.1 s -> ~1.3-1.6 s per 8K chunk (+15-20% prefill), flatter 32K ramp.
2. GDN prompt recurrence (gdn_rec_cols_pipe_kernel: token-serial FP32, 48 heads x 4 column blocks, state slice in
   registers, next token prefetched) + gdn_conv/l2/gates/out_norm. sycl/src/prefill/kernels.dp.cpp. Expected 0.9-2 s -> ~0.4-0.5 s per 8K.
3. MTP self-speculation (largest decode lever): model's own MTP layer, window 4-6, confidence gate 0.5, reduced-vocab
   draft head, MTP experts low-bit in VRAM. Expected bs1 decode ~22 -> 45-60 tok/s. Via SGLang speculative path +
   exl3xpu verify for T>1 + GDN state rollback. Hard.
4. RAM tier as a USM-host mirror read zero-copy by the kernels (their 1,879 mirrored experts cost only -6%), instead of
   our memfd+SVM tier. Prime suspect for our unexplained ~10-13 ms/step decode tier overhead. Seed VRAM/RAM ranking
   from their data/expert-profile.bin (same expert ids) + adaptive LFU swaps (every 4 rounds, up to 96, decay 0.7,
   admit only after the copy lands).
5. Decode non-MoE fusions + whole-window SYCL graph with device step buffers: fused_gr_read(_multi) (HC read 6->2
   kernels), fused_gdn_conv_l2 / fused_gdn_ab / fused_gdn_step_norm, decode QSA. Target non-MoE decode 25 -> 8-12 ms/step.
   Also: aligned wide loads (misaligned 16-B loads split on B70; 2.3-4.7x per kernel for them), fewer graph nodes (~5 us each).
6. Prefill hygiene: 256-token first chunk (PLE rows load while GPU starts; 857 -> 986 tok/s for them), next-chunk PLE
   prefetch, grouping tables in mapped memory, sequence-number polling instead of host event waits (Level Zero v2
   deadlock), borrow expert-cache slots for prefill buffers (790 -> 1,062 @80K).

## Do not copy
- CPU computing cold experts concurrently via host-mapped doorbells: on Intel, host flag stores are not reliably
  visible to running kernels (B60: 11.9 tok/s, 187 ms ring waits/round). Their own intra-layer overlap was -7% on NVIDIA.
- joint_matrix XMX kernels for decode experts / prompt attention (1.4-5x slower on B70); their prefill MoE
  (~1,000 oneMKL launches/layer + per-layer host sync).

## Arc pitfalls they document
VRAM over-allocation on xe evicts to host and can livelock the box (keep >=1.5 GB free); L0 v2 event-wait deadlock
(poll a seq number or SYCL_UR_USE_LEVEL_ZERO_V2=0); unbounded device spins reset/wedge the GT (bound every spin);
SYCL_CACHE_PERSISTENT=1 segfaults on Xe2; dpct memcpy does not block; sub-group 16 races in warp-32 code;
-cl-fp32-correctly-rounded-divide-sqrt for exact divides.

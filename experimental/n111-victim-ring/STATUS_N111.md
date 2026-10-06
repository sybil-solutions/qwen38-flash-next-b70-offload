N111: asynchronous victim write-back through a VRAM ring (B70 NVMe-tier MoE decode). Final status 2026-10-06 23:45 CEST.
The work stopped at the standalone stage when priority moved to GLM-5.3-Flash on the 3090. The srv27 serve matrix was
SKIPPED, so there are no end-to-end server numbers.

1. Problem
The decode path moe_forward_cached evicts a VRAM slot on a miss. In tier mode it then writes the victim (1,862,400 B)
back to its RAM-tier page through SVM GPU stores (about 0.9-1.0 ms per victim, N109). This happens in order, before
the GEMV, and costs about 10-13 ms per step. With NOVICT the decode cost disappears, but masked picks rise about 4x
because evicted experts are lost from RAM.

2. Design (implemented)
Kernel a9 (csrc/exl3_moe.sycl, diff a9.patch against the a8 source exl3_moe.a8.sycl):
- moe_set_victim_ring(meta int32[4+2K], sel int32[E], buf u8[K*blob]).
  * meta[0] = next search position, [1] = claim sequence, [2] = drops, [3] = K.
  * key[K]: -1 = free, else the victim key. seq[K]: the claim sequence of that key.
- The ensure kernel (work-item 0, after the assign barrier) walks the victims of the call and claims a free slot
  (key == -1) for each one, writing key, seq and sel[i].
- When the ring is full the victim is dropped: no write-back, the same as NOVICT, counted in meta[2]. I chose this over
  a fallback direct store because the store is the ~1 ms per victim being removed. Running out of ring means the drain
  can't keep up, and degrading to NOVICT keeps the step time flat.
- VictimCopyKernel in ring mode copies the VRAM slot to the VRAM ring slot (D2D) and does not set ram_res.
- The capture-time globals are baked into the decode graph. The setter is called at tier init, before capture.
Kernel a9b (csrc/exl3_moe.a9b.sycl, a9b.patch) = a9 + a native drain thread with its own in-order SYCL queue
(moe_vring_drain_push / _done / _stop).
Libraries:
- _moe_a9.so and _moe_a9b.so are in this dir. _moe_a9.so is also in runs/N104-nvtier/src/exl3xpu/; _moe_a8.so there is
  untouched.
- The a8 source rebuilt to the same size as _moe_a8.so, within 8 B.

Host: class NvTierAsyncRing(NvTierAsync) in nvtier.py (this dir).
- Each step it reads the one-step-lagged snapshot of the ring meta together with res/miss-ring/slot_of.
- A claim is new when key >= 0 and seq != the seq already handled for that slot.
- Each new claim gets a copy to its memfd page. Drain modes (EXL3_NVTIER_VRING_DRAIN):
  * "usm" (default): async copy-engine D2H into a pinned USM staging slot, then a CPU thread memmoves staging to the
    memfd page.
  * "stream": D2H on a side stream straight into the memfd page.
  * "thread": the a9b native worker.
- Only after the copy has completed does the host set res[key]=1, through the normal diff upload. It then frees the ring
  slot by uploading key=-1 on the compute stream, which is ordered before the next replay.
Safety invariants:
- A ring slot is reused only after its copy completed: the device claims only key==-1 slots, and only the host writes -1.
- The resident flag is set only after the data landed. The device never sets the flag in ring mode.
- vr_busy: a key with a write-back in flight is never an eviction or punch candidate. This holds in the step budget and
  in _ensure_for_prefill. That is the only change to base code, and it does nothing when vr_busy is absent.
- ppend: a write-back to a page whose punch is still pending (queued or in _clears) is deferred to a later step.
- Misses on busy keys skip the NVMe fill, because the data is about to land.
- Budget counts in-flight write-backs.
- Stats line: "nvtier vring last N steps: claims/drops/issued/landed/deferred per step".
Plugin and serve wiring:
- sglang_plugin.tier.py (this dir): EXL3_NVTIER_VICTIM_RING=1 selects NvTierAsyncRing. It needs EXL3_NVTIER_ASYNC=1.
- runs/N104-nvtier/n104_serve.sh knob: VRING=1 mounts the a9 lib and the N111 nvtier.py/plugin and sets
  EXL3_NVTIER_VICTIM_RING=1, EXL3_NVTIER_VRING_K=${VRK:-128} and EXL3_NVTIER_DECODE_PF=${DECPF:-0}.
  * The default docker command (knob unset) is byte-identical to before; checked with a dry run.
  * Backup: n104_serve.sh.pre_n111.
  * The N104 nvtier.py and plugin are not modified.
  * TODO before using it: switch MOELIB to _moe_a9b.so if the "thread" drain is wanted. "usm" works with a9.

3. Results (standalone, B70 0000:84:00.0, image 24c872759256; n111_test.py, run.sh / run_py.sh, guard_boot.sh)
exact (logs x9 / x8 / s9b / s9u): 72 calls at M=1/2/4 with 396 victims, 4 real layers from the NVMe store.
- Outputs == golden all-VRAM moe_forward in plain, wb and ring mode: 72/72 each.
- a8 and a9/a9b outputs are byte-identical (exact_*.npy; cmp of plain/wb/ring).
- Ring: 396/396 ring slots hold the victim data, ram_res stays 0 and the RAM page is untouched.
sim (TierStore + classes, 4 layers, 640 VRAM slots, 640 RAM experts, trace routing, M=1, 1200 steps):
- ring K=128: 0 mismatched output rows (every call checked against golden with the post-mask ids).
  * 3,157 pages checked at the moment the host flagged them resident: 0 bad.
  * 3,072 audited pages: 0 bad. Ring slots: 152/152 ok.
  * masked/step 2.7-4.1, 2.65 claims/step, 0 drops.
- ring K=16: 0 mismatches, about 0.4-0.5 drops/step, masked/step 4.0-5.0.
- wb (current direct write-back, NvTierAsync): 50-52 mismatched rows and 8-10 of about 3,070 audited pages bad, in
  two runs. masked/step 2.5-3.7.
- novict: masked/step 32.9.
- At M=4 the sim's RAM tier is too small: about 150 masked/step in every mode, so it is not informative.

D2H blocking (d2h2): 30 x 1.86 MB from VRAM.
- To USM host: issue 0.18 ms, done in 6.3 ms (8.9 GB/s).
- To memfd (warm or punched): issue == total = 9.4-14.8 ms. The submitting thread blocks for the whole copy
  (about 0.41 ms per expert, 4.5 GB/s).

bench: 48 decode calls (M=1) per step, write-back targets freshly punched. GPU ms per step minus the no-victim cost:

| victims/step | wb (current) | ring + side-stream D2H | ring + a9b thread |
|---|---|---|---|
| 10 | +8.6 to +11.1 | +0.2 | +0.5 to +1.3 |
| 20 | +20.0 to +25.1 | +0.1 to +0.3 | +2.1 |
| 30 | +29.3 to +29.9 | +0.4 to +0.5 | +4.4 |

- Host wall per step versus none: the side-stream drain blocks the submitting host thread for about 0.45 ms per victim
  (13.8 ms at 30). The a9b thread drain issues in about 0.1 ms, but the concurrent pageable copies slow the GPU stream
  by about 0.15 ms per victim.
- The USM-staging drain (ringU) bench was NOT measured: a test-only bug (the ringU branch also fell into the stream
  branch) crashed the last bench case. The class's "usm" drain itself passed the sim above with 0 mismatches.

4. Incidents
- 21:39: the B70 84 fell off the bus during n110-qa-A. My last GPU job before that had ended at 21:29.
  * My queued n111-s9b then took the lock on the re-enumerated renderD132 and ran (21:49, completed). The old guard
    read the kernel log only from job start, so it missed the 21:39 lines.
  * Fixed with guard_boot.sh: checks journalctl -k -b since boot for Completion-Wait / Link Down / Card not present /
    reboot is needed / non-corrected GHES, /health 200, no D-state khugepaged/kcompactd, and 84:00.0 == renderD130.
- 23:43: the s9u container was left running after its python had crashed, because my ssh session was cut by an alarm
  before run.sh's docker stop. I stopped it with docker stop n111-test.
  * The post guard was KERNEL_OK and DSTATE_OK, but health=000: dsv41-lab is not running (not touched by me). It was
    gone at 23:45, and it had been running at 23:21.

5. Next steps (for the B70 tier, or ported to the GLM tier)
- Run the ringU bench and switch the default drain to "usm". The expected cost is about 0.2-0.5 ms per step of GPU time
  (the D2D copy) with no host blocking.
- Then run the srv27 matrix with VRING=1 (command in the task) and compare with srv23/srv26.
- Fix or retire the direct write-back race in the current path:
  * The device sets ram_res=1 after the host's lagged snapshot while the host's punch of that key is still queued, so
    the page is zeroed while flagged resident.
  * The ring design avoids this because the host owns every flag transition. The same rule applies to any GLM tier:
    the device never publishes residency.

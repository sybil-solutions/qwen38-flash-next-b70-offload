"""N104 NVMe expert tier for the B70 device-managed expert cache (exl3xpu, kernel patches a2/a3/a5).

Tiers: VRAM slots (device LRU, write-through on miss) -> RAM tier (sparse memfd, one 2 MiB huge page per expert at a
fixed address, read by the GPU through xe SVM) -> NVMe store (packed 4K-aligned records, O_DIRECT).

Device side (kernel): picks whose expert is in neither VRAM nor RAM are masked to (safe expert, weight 0) and logged to a
miss ring; VRAM victims are written back to their RAM address and flagged resident (exclusive RAM tier).
Host side (this file, call step() between decode steps): fill logged misses from NVMe, keep the RAM tier under budget
(evict RAM copies of VRAM-resident experts first, then oldest), flags updated on the device before any page is punched.
"""
import ctypes, json, mmap, os, time
from collections import OrderedDict
import numpy as np
import torch
from .moe_offload import ExpertStore, ops, s64

MiB2 = 2 << 20
libc = ctypes.CDLL(None, use_errno=True)
FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 1, 2


class TierStore(ExpertStore):
    """ExpertStore whose host copies live at fixed 2 MiB-stride addresses in a sparse memfd region (not USM)."""

    def __init__(self, H, I, K, E, n_slots, n_layers, stride=MiB2):
        super().__init__(H, I, K, E, n_slots, max_layers=n_layers)
        assert self.blob <= stride
        self.stride, self.L = stride, n_layers
        self.size = n_layers * E * stride
        self.fd = os.memfd_create("nvtier", 0)
        os.ftruncate(self.fd, self.size)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        resv = libc.mmap(None, self.size + MiB2, 0, 0x22, -1, 0)                  # PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS
        aligned = (resv + MiB2 - 1) // MiB2 * MiB2
        b = libc.mmap(ctypes.c_void_p(aligned), self.size, 3, 0x01 | 0x10, self.fd, 0)   # RW, MAP_SHARED|MAP_FIXED
        assert b == aligned, (hex(b or 0), hex(aligned))
        self.base = aligned
        libc.madvise(ctypes.c_void_p(self.base), ctypes.c_size_t(self.size), 14)      # MADV_HUGEPAGE
        self.X.moe_set_host_stride(stride)

    def add_layer(self, key, blobs=None):
        li = len(self.layer_index)
        self.layer_index[key] = li
        base = s64(self.base + li * self.E * self.stride)
        self._host_ptr[key] = base + torch.arange(self.E, dtype=torch.int64) * self.stride
        self.ptrs_all[li * self.E:(li + 1) * self.E].copy_(self._host_ptr[key])
        self.host_base[li] = base
        self.slot_of[key] = torch.full((self.E,), -1, dtype=torch.int32)
        return None

    def addr(self, key):                       # key = li * E + e
        return self.base + key * self.stride


class NvTier:
    def __init__(self, store: TierStore, nvme_path, meta, ram_budget_bytes, ring_cap=1 << 16, safe=0, dev=None, ckpt_layer=None):
        self.st, self.E = store, store.E
        self.L = len(ckpt_layer) if ckpt_layer is not None else store.L
        self.ckpt = np.asarray(ckpt_layer if ckpt_layer is not None else range(self.L), dtype=np.int64)   # li -> NVMe layer
        self.dev = dev or store.dev
        self.blob, self.rec = int(meta["blob"]), int(meta["rec"])
        assert self.blob == store.blob, (self.blob, store.blob)
        self.budget = max(1, int(ram_budget_bytes // store.stride))      # experts that may be RAM-resident
        self.nfd = os.open(nvme_path, os.O_RDONLY | os.O_DIRECT)
        self.buf = mmap.mmap(-1, self.rec)                                # page-aligned bounce buffer for O_DIRECT
        self.bufp = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        n = self.L * self.E
        self.res_dev = torch.zeros(n, dtype=torch.uint8, device=self.dev)
        self.ring = torch.zeros(1 + ring_cap, dtype=torch.int32, device=self.dev)
        self.safe = torch.full((self.L,), safe, dtype=torch.int32, device=self.dev)
        self.vict = torch.zeros(1 + 2 * self.E, dtype=torch.int32, device=self.dev)
        self.res = np.zeros(n, dtype=np.uint8)                            # host view (refreshed each step)
        self.order = OrderedDict()                                        # RAM-resident keys, oldest first
        self.pinned = set(li * self.E + safe for li in range(self.L))
        self.ring_seen = 0
        self.pending_evict = []                                           # (event, keys) waiting before punch
        self.stats = dict(nvme_reads=0, nvme_ms=0.0, evicts=0, masked=0, steps=0, fill_ms=0.0, ring_overflow=0)
        for k in sorted(self.pinned):
            self._fill(k)
        self.res_dev.copy_(torch.from_numpy(self.res))
        X = store.X
        X.moe_set_tier(self.res_dev, self.ring, self.safe)
        if os.environ.get("EXL3_NVTIER_NO_VICTIM", "0") != "1":
            X.moe_set_victims(self.vict)   # VRAM victims written back to the RAM tier by the decode kernel (~0.9 ms each, N109)
        else:
            X.moe_set_victims(torch.zeros(0, dtype=torch.int32, device=self.dev))   # N110: no write-back; misses refill from NVMe

    def close(self):
        X = self.st.X
        X.moe_set_tier(torch.zeros(0, dtype=torch.uint8, device=self.dev), torch.zeros(0, dtype=torch.int32, device=self.dev),
                       torch.zeros(0, dtype=torch.int32, device=self.dev))
        X.moe_set_victims(torch.zeros(0, dtype=torch.int32, device=self.dev))
        X.moe_set_host_stride(0)

    def _fill(self, key):
        t = time.perf_counter()
        n = os.preadv(self.nfd, [self.buf], self.rec_index(key) * self.rec)
        assert n == self.rec, (n, self.rec)
        ctypes.memmove(self.st.addr(key), self.bufp, self.blob)
        self.res[key] = 1
        self.order[key] = None
        self.stats["nvme_reads"] += 1
        self.stats["nvme_ms"] += (time.perf_counter() - t) * 1e3

    def rec_index(self, key):
        return int(self.ckpt[key // self.E]) * self.E + key % self.E

    def _punch(self, key):
        off = key * self.st.stride
        r = libc.fallocate(self.st.fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, ctypes.c_long(off), ctypes.c_long(self.st.stride))
        assert r == 0, ctypes.get_errno()

    def step(self):
        """Between decode steps (device work of the step already queued): fill misses, enforce budget."""
        t0 = time.perf_counter()
        torch.xpu.synchronize()                    # simple + exact: everything queued so far has finished
        res_d = self.res_dev.cpu().numpy()
        # 1) punches whose flag-clear has been observed by all earlier work (we just synced). A key the GPU wrote back
        #    (victim) in the meantime is flagged 1 again: keep it, never punch live data.
        for keys in self.pending_evict:
            for k in keys:
                if res_d[k]:
                    self.stats["evict_rescued"] = self.stats.get("evict_rescued", 0) + 1
                else:
                    self._punch(k)
        self.pending_evict = []
        # 2) device -> host: residency flags (victims flagged by the device) and VRAM residency
        newly = np.nonzero((res_d == 1) & (self.res == 0))[0]
        for k in newly.tolist():
            self.order[k] = None                    # victims written back by the GPU become the newest RAM entries
        self.res = res_d.copy()
        cnt = int(self.ring[0])
        newm = cnt - self.ring_seen
        if newm > self.ring.numel() - 1:
            self.stats["ring_overflow"] += newm - (self.ring.numel() - 1); newm = self.ring.numel() - 1
        keys = []
        if newm > 0:
            cap = self.ring.numel() - 1
            ring = self.ring.cpu().numpy()
            keys = [int(ring[1 + (i % cap)]) for i in range(cnt - newm, cnt)]
        self.ring_seen = cnt
        self.stats["masked"] += newm
        # 3) fill misses from NVMe (dedup, skip already resident)
        for k in dict.fromkeys(keys):
            if not self.res[k]:
                self._fill(k)
        # 4) budget: evict RAM copies of VRAM-resident experts first, then the oldest
        over = int(self.res.sum()) - self.budget
        ev = []
        if over > 0:
            vram = self.st.slot_of_dev.cpu().numpy()[: self.L * self.E] >= 0
            for k in list(self.order.keys()):
                if over <= 0: break
                if self.res[k] and vram[k] and k not in self.pinned:
                    ev.append(k); over -= 1
            for k in list(self.order.keys()):
                if over <= 0: break
                if self.res[k] and k not in self.pinned and k not in ev:
                    ev.append(k); over -= 1
            for k in ev:
                self.res[k] = 0; self.order.pop(k, None)
            self.stats["evicts"] += len(ev)
            self.pending_evict.append(ev)          # punched at the next step(), after a sync
        # 5) host -> device flags (fills + evictions)
        self.res_dev.copy_(torch.from_numpy(self.res))
        self.stats["steps"] += 1
        self.stats["fill_ms"] += (time.perf_counter() - t0) * 1e3
        return len(keys)

    def ram_bytes(self):
        return int(self.res.sum()) * self.st.stride


class NvTierFast(NvTier):
    """c3: vectorized bookkeeping + background I/O thread (NVMe fills, hole punches). Per-token host work = one sync,
    small D2H/H2D of flags, numpy selection. Fills become visible one step later (misses stay masked until then)."""

    def __init__(self, *a, **k):
        import threading, queue
        super().__init__(*a, **k)
        n = self.L * self.E
        self.age = np.zeros(n, dtype=np.int64); self.tick = 1
        for key in self.order: self.age[key] = 0
        self.pin_mask = np.zeros(n, dtype=bool); self.pin_mask[list(self.pinned)] = True
        self.inflight = np.zeros(n, dtype=bool)          # fill requested, not yet done
        self.q = queue.Queue(); self.done = queue.Queue()
        self.th = threading.Thread(target=self._worker, daemon=True); self.th.start()

    def _worker(self):
        buf = mmap.mmap(-1, self.rec); bufp = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        while True:
            op, key = self.q.get()
            if op == "fill":
                t = time.perf_counter()
                n = os.preadv(self.nfd, [buf], self.rec_index(key) * self.rec); assert n == self.rec
                ctypes.memmove(self.st.addr(key), bufp, self.blob)
                self.done.put(key)
                self.stats["nvme_reads"] += 1; self.stats["nvme_ms"] += (time.perf_counter() - t) * 1e3
            elif op == "punch":
                # a prefill fill may have re-populated the key since it was queued: never punch resident data
                if not getattr(self, "no_punch", False) and not self.res[key]: self._punch(key)
            elif op == "stop":
                return

    def step(self):
        t0 = time.perf_counter()
        self.tick += 1
        if getattr(self, "_t_end", None) is not None:
            self.stats["outside_ms"] = self.stats.get("outside_ms", 0.0) + (t0 - self._t_end) * 1e3   # sglang + replay launch
        torch.xpu.synchronize()
        t_s = time.perf_counter()
        self.stats["sync_ms"] = self.stats.get("sync_ms", 0.0) + (t_s - t0) * 1e3                  # GPU still busy
        res_d = self.res_dev.cpu().numpy()
        # punches queued last step: rescue keys the GPU wrote back since (flag 1 again), punch the rest in the background
        for keys in self.pending_evict:
            for k in keys:
                if res_d[k]: self.stats["evict_rescued"] = self.stats.get("evict_rescued", 0) + 1
                else: self.q.put(("punch", k))
        self.pending_evict = []
        newly = (res_d == 1) & (self.res == 0)
        self.age[newly] = self.tick                       # victims written back by the GPU = newest RAM entries
        if hasattr(self, "transient"): self.transient[newly] = False
        res = res_d.copy()
        # completed fills -> resident
        changed_on = []
        while not self.done.empty():
            k = self.done.get(); res[k] = 1; self.age[k] = self.tick; self.inflight[k] = False; changed_on.append(k)
        # new misses from the ring -> queue fills (dedup)
        cnt = int(self.ring[0]); newm = cnt - self.ring_seen
        cap = self.ring.numel() - 1
        if newm > cap: self.stats["ring_overflow"] += newm - cap; newm = cap
        if newm > 0:
            lo = (cnt - newm) % cap; hi = cnt % cap
            r = self.ring[1:].cpu().numpy()
            keys = np.concatenate([r[lo:], r[:hi]]) if hi <= lo and newm > 0 else r[lo:hi]
            keys = np.unique(keys[:newm])
            keys = keys[(res[keys] == 0) & (~self.inflight[keys])]
            for k in keys.tolist():
                self.inflight[k] = True; self.q.put(("fill", k))
        self.ring_seen = cnt; self.stats["masked"] += max(0, newm)
        # budget: RAM copies of VRAM-resident experts first (oldest), then oldest overall
        over = int(res.sum()) + int(self.inflight.sum()) - self.budget
        ev = np.zeros(0, dtype=np.int64)
        if over > 0:
            vram = self.st.slot_of_dev[: self.L * self.E].cpu().numpy() >= 0
            if not getattr(self, "dup_first", True):
                vram = vram & False                      # policy B: plain oldest-first, duplicates are kept
            cand1 = np.nonzero((res == 1) & vram & ~self.pin_mask)[0]
            take1 = cand1[np.argsort(self.age[cand1])[:over]]
            over -= len(take1)
            take2 = np.zeros(0, dtype=np.int64)
            if over > 0:
                m = (res == 1) & ~self.pin_mask; m[take1] = False
                cand2 = np.nonzero(m)[0]
                take2 = cand2[np.argsort(self.age[cand2])[:over]]
            ev = np.concatenate([take1, take2])
            res[ev] = 0
            self.stats["evicts"] += len(ev)
            self.pending_evict.append(ev.tolist())
        # host -> device: only the changed flags
        diff = np.nonzero(res != res_d)[0]
        if len(diff):
            idx = torch.from_numpy(diff.astype(np.int64)).to(self.dev, non_blocking=True)
            val = torch.from_numpy(res[diff]).to(self.dev, non_blocking=True)
            self.res_dev.index_copy_(0, idx, val)
        self.res = res
        self.stats["steps"] += 1
        self._t_end = time.perf_counter()
        self.stats["fill_ms"] += (self._t_end - t0) * 1e3
        return int(max(0, newm))

    def ram_bytes(self):
        return int(self.res.sum()) * self.st.stride

    def close(self):
        self.q.put(("stop", 0)); self.th.join(timeout=10)
        super().close()



class NvTierAsync(NvTierFast):
    """n107: NvTierFast without the per-step torch.xpu.synchronize(); one-step-lagged, event-ordered bookkeeping."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n = self.L * self.E
        self._n = n
        self._h_res = [torch.empty(n, dtype=torch.uint8).pin_memory() for _ in range(2)]
        self._h_ring = [torch.empty(self.ring.numel(), dtype=torch.int32).pin_memory() for _ in range(2)]
        self._h_slot = [torch.empty(n, dtype=torch.int32).pin_memory() for _ in range(2)]
        self._snap_ev = [None, None]
        self._k = 0
        self._last_diff = None            # (keys int64 np, vals uint8 np) uploaded by the previous step
        self._clears = []                 # [(event, np keys)] evicted keys awaiting their flag-clear completion

    def step(self):
        t0 = time.perf_counter()
        self.tick += 1
        if getattr(self, "_t_end", None) is not None:
            self.stats["outside_ms"] = self.stats.get("outside_ms", 0.0) + (t0 - self._t_end) * 1e3
        n = self._n
        cur = torch.xpu.current_stream()
        k = self._k; self._k ^= 1
        # 1) snapshot of the device state after this step's replay (consumed next step)
        self._h_res[k].copy_(self.res_dev[:n], non_blocking=True)
        self._h_ring[k].copy_(self.ring, non_blocking=True)
        self._h_slot[k].copy_(self.st.slot_of_dev[:n], non_blocking=True)
        ev = torch.xpu.Event(); ev.record(cur); self._snap_ev[k] = ev
        j = k ^ 1
        if self._snap_ev[j] is None:
            self._t_end = time.perf_counter(); self.stats["steps"] += 1
            return 0
        t_s = time.perf_counter()
        self._snap_ev[j].synchronize()    # normally already complete (queued before this step's replay)
        self.stats["sync_ms"] = self.stats.get("sync_ms", 0.0) + (time.perf_counter() - t_s) * 1e3
        res_d = self._h_res[j].numpy().copy()
        if self._last_diff is not None:
            dk, dv = self._last_diff
            res_d[dk] = dv                # device state after the previous step's flag upload
        ring = self._h_ring[j].numpy()
        vram = self._h_slot[j].numpy() >= 0
        # 2) punch evicted keys whose flag clear has completed on the device
        keep = []
        for ce, keys in self._clears:
            if ce.query():
                for kk in keys.tolist():
                    if res_d[kk]:
                        self.stats["evict_rescued"] = self.stats.get("evict_rescued", 0) + 1
                    else:
                        self.q.put(("punch", kk))
            else:
                keep.append((ce, keys))
        self._clears = keep
        # 3) victims the GPU wrote back since (flag 1 on device, 0 on host) become the newest RAM entries
        newly = (res_d == 1) & (self.res == 0)
        self.age[newly] = self.tick
        if hasattr(self, "transient"): self.transient[newly] = False
        res = self.res.copy()
        res[newly] = 1
        while not self.done.empty():
            kk = self.done.get(); res[kk] = 1; self.age[kk] = self.tick; self.inflight[kk] = False
        # 4) new misses from the ring snapshot -> background fills
        cnt = int(ring[0]); newm = cnt - self.ring_seen
        cap = self.ring.numel() - 1
        if newm > cap: self.stats["ring_overflow"] += newm - cap; newm = cap
        if newm > 0:
            lo = (cnt - newm) % cap; hi = cnt % cap
            r = ring[1:]
            keys = np.concatenate([r[lo:], r[:hi]]) if hi <= lo else r[lo:hi]
            keys = np.unique(keys[:newm]).astype(np.int64)
            keys = keys[(res[keys] == 0) & (~self.inflight[keys])]
            for kk in keys.tolist():
                self.inflight[kk] = True; self.q.put(("fill", kk))
        self.ring_seen = cnt; self.stats["masked"] += max(0, newm)
        # 5) budget: RAM copies of VRAM-resident experts first (oldest), then oldest overall
        over = int(res.sum()) + int(self.inflight.sum()) - self.budget
        ev_keys = np.zeros(0, dtype=np.int64)
        if over > 0:
            cand1 = np.nonzero((res == 1) & vram & ~self.pin_mask)[0]
            take1 = cand1[np.argsort(self.age[cand1])[:over]]
            over -= len(take1)
            take2 = np.zeros(0, dtype=np.int64)
            if over > 0:
                m = (res == 1) & ~self.pin_mask; m[take1] = False
                cand2 = np.nonzero(m)[0]
                take2 = cand2[np.argsort(self.age[cand2])[:over]]
            ev_keys = np.concatenate([take1, take2]).astype(np.int64)
            res[ev_keys] = 0
            self.stats["evicts"] += len(ev_keys)
        # 6) host -> device: changed flags only (pinned staging, async), then the clear event for evicted keys
        diff = np.nonzero(res != res_d)[0].astype(np.int64)
        if len(diff):
            idx = torch.from_numpy(diff).pin_memory().to(self.dev, non_blocking=True)
            val = torch.from_numpy(res[diff].copy()).pin_memory().to(self.dev, non_blocking=True)
            self.res_dev.index_copy_(0, idx, val)
            self._last_diff = (diff, res[diff].copy())
        else:
            self._last_diff = None
        if len(ev_keys):
            ce = torch.xpu.Event(); ce.record(cur); self._clears.append((ce, ev_keys))
        self.res = res
        self.stats["steps"] += 1
        self._t_end = time.perf_counter()
        self.stats["fill_ms"] += (self._t_end - t0) * 1e3
        return int(max(0, newm))


def _ensure_for_prefill(self, li, ids):
    """Prefill (eager): synchronously make every expert this layer's batch uses RAM-resident (NVMe fill), flag it
    resident on the device, so the prefill kernel can read it zero-copy.
    d1b: the RAM budget holds during prefill. Before filling, sync the device (no queued kernel still reads a page we
    punch), then evict: prefill-only (transient) fills of other layers first, then RAM copies of VRAM-resident experts,
    then the oldest; never this layer's keys, pinned keys or in-flight decode fills. Flags cleared before the punch."""
    n = self.L * self.E
    T0 = time.perf_counter()
    if not hasattr(self, "transient"): self.transient = np.zeros(n, dtype=bool)
    keys = np.unique(ids.reshape(-1).to("cpu", torch.int64).numpy()) + li * self.E
    miss = keys[self.res[keys] == 0]
    if len(miss):
        inflight = getattr(self, "inflight", np.zeros(n, dtype=bool))
        over = int(self.res.sum()) + int(inflight.sum()) + len(miss) - self.budget
        if over > 0:
            t = time.perf_counter()
            torch.xpu.synchronize()
            res_d = self.res_dev.cpu().numpy()
            newly = (res_d == 1) & (self.res == 0)            # victims the GPU wrote back since the last step
            self.res[newly] = 1
            if hasattr(self, "age"): self.age[newly] = self.tick
            age = getattr(self, "age", np.zeros(n, dtype=np.int64))
            ok = (self.res == 1) & ~inflight
            ok[list(self.pinned)] = False; ok[keys] = False
            vram = self.st.slot_of_dev[:n].cpu().numpy() >= 0
            ev = []
            for m in (ok & self.transient, ok & ~self.transient & vram, ok & ~self.transient & ~vram):
                if over <= 0: break
                c = np.nonzero(m)[0]; c = c[np.argsort(age[c], kind="stable")[:over]]
                ev.append(c); over -= len(c)
            ev = np.concatenate(ev) if ev else np.zeros(0, dtype=np.int64)
            if len(ev):
                self.res[ev] = 0; self.transient[ev] = False
                self.res_dev.index_fill_(0, torch.from_numpy(ev.astype(np.int64)).to(self.dev), 0)
                torch.xpu.synchronize()
                for k in ev.tolist(): self._punch(k)
                if hasattr(self, "order"):
                    for k in ev.tolist(): self.order.pop(k, None)
            self.stats["prefill_evicts"] = self.stats.get("prefill_evicts", 0) + len(ev)
            self.stats["prefill_evict_ms"] = self.stats.get("prefill_evict_ms", 0.0) + (time.perf_counter() - t) * 1e3
            if over > 0: self.stats["prefill_over_budget"] = self.stats.get("prefill_over_budget", 0) + over
    miss = [int(k) for k in miss]
    T1 = time.perf_counter()
    if os.environ.get("EXL3_NVTIER_PAR_FILL", "1") == "1": self.fill_many(miss)
    else:
        for k in miss: self._fill(k)
    for k in miss:
        if hasattr(self, "inflight"): self.inflight[k] = False
        if hasattr(self, "age"): self.age[k] = self.tick
        self.transient[k] = True
    T2 = time.perf_counter()
    if miss:
        if os.environ.get("EXL3_NVTIER_SVM_PREFETCH", "1") == "1" and hasattr(self.st.X, "moe_svm_prefetch"):
            # map the freshly filled pages on the GPU side before the kernel touches them (a8)
            self.st.X.moe_svm_prefetch(torch.tensor([self.st.addr(k) for k in miss], dtype=torch.int64), self.st.stride, self.res_dev)
        idx = torch.tensor(miss, dtype=torch.int64).to(self.dev, non_blocking=True)
        self.res_dev.index_fill_(0, idx, 1)
    self.stats["prefill_fills"] = self.stats.get("prefill_fills", 0) + len(miss)
    T3 = time.perf_counter()
    P = self.__dict__.setdefault("_prof", dict(calls=0, pre_ms=0.0, fill_ms=0.0, post_ms=0.0, gap_ms=0.0, fills=0, last=None))
    if P["last"] is not None: P["gap_ms"] += (T0 - P["last"]) * 1e3     # time between calls = GPU/other work of the layer
    P["calls"] += 1; P["pre_ms"] += (T1 - T0) * 1e3; P["fill_ms"] += (T2 - T1) * 1e3; P["post_ms"] += (T3 - T2) * 1e3
    P["fills"] += len(miss); P["last"] = T3
    if P["calls"] % int(os.environ.get("EXL3_NVTIER_PROF_EVERY", "48")) == 0:
        import logging
        logging.getLogger("exl3xpu").warning("nvtier prefill prof: calls %d fills %d (%.1f GB) pre(evict) %.0f ms fill %.0f ms (%.1f GB/s) post(prefetch+flags) %.0f ms between-calls %.0f ms",
            P["calls"], P["fills"], P["fills"] * self.rec / 1e9, P["pre_ms"], P["fill_ms"], P["fills"] * self.rec / 1e6 / max(P["fill_ms"], 1e-3), P["post_ms"], P["gap_ms"])
        P.update(calls=0, pre_ms=0.0, fill_ms=0.0, post_ms=0.0, gap_ms=0.0, fills=0)
    return len(miss)


def _fill_many(self, keys):
    """d1b2: parallel O_DIRECT reads straight into the 2 MiB-aligned memfd slots (no bounce buffer, no memmove).
    The RAID0 store needs queue depth >= 8-16 to reach ~21-26 GB/s; one record per call keeps each read 4K-aligned."""
    keys = [int(k) for k in keys]
    if not keys: return
    if len(keys) < 4:
        for k in keys: self._fill(k)
        return
    if not hasattr(self, "_pool"):
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(int(os.environ.get("EXL3_NVTIER_FILL_THREADS", "32")))
    rec, nfd = self.rec, self.nfd
    assert rec <= self.st.stride and rec % 4096 == 0
    def rd(k):
        buf = (ctypes.c_char * rec).from_address(self.st.addr(k))
        n = os.preadv(nfd, [buf], self.rec_index(k) * rec)
        assert n == rec, (n, rec)
    t = time.perf_counter()
    list(self._pool.map(rd, keys))
    for k in keys:
        self.res[k] = 1
        if hasattr(self, "order"): self.order[k] = None
    self.stats["nvme_reads"] += len(keys)
    self.stats["nvme_ms"] += (time.perf_counter() - t) * 1e3


NvTier.fill_many = _fill_many


NvTier.ensure_for_prefill = _ensure_for_prefill


def _warm(self, keys_ckpt, max_bytes):
    """Fill the RAM tier with the hottest experts (checkpoint keys layer*E+e, hottest first) up to max_bytes."""
    inv = {int(c): li for li, c in enumerate(self.ckpt)}
    n = 0; t = time.perf_counter(); cap = int(max_bytes // self.st.stride)
    have = int(self.res.sum()); todo = []; seen = set()
    for kc in keys_ckpt:
        if have + len(todo) >= cap: break
        li = inv.get(int(kc) // self.E)
        if li is None: continue
        k = li * self.E + int(kc) % self.E
        if not self.res[k] and k not in seen: todo.append(k); seen.add(k)
    self.fill_many(todo); n = len(todo)
    if hasattr(self, "age"): self.age[todo] = 0
    self.res_dev.copy_(torch.from_numpy(self.res))
    return n, time.perf_counter() - t


NvTier.warm = _warm


def _log_stats(self, logger, every=200):
    if self.stats["steps"] % every or not self.stats["steps"]:
        return
    b = getattr(self, "_last", None) or {k: 0 for k in self.stats}
    d = {k: self.stats.get(k, 0) - b.get(k, 0) for k in self.stats}
    n = max(1, d.get("steps", 0))
    logger.info("exl3xpu: nvtier last %d steps: masked/step %.2f nvme/step %.2f evicts/step %.1f host ms/step %.2f "
                "prefill fills %d evicts %d (%.0f ms) rescued %d ram %.1f GB (memfd allocated %.1f GB) | per step: outside %.1f ms, sync-wait %.1f ms, host work %.1f ms", n, d.get("masked", 0) / n,
                d.get("nvme_reads", 0) / n, d.get("evicts", 0) / n, d.get("fill_ms", 0) / n, d.get("prefill_fills", 0),
                d.get("prefill_evicts", 0), d.get("prefill_evict_ms", 0), d.get("evict_rescued", 0), self.ram_bytes() / 2 ** 30, os.fstat(self.st.fd).st_blocks * 512 / 2 ** 30,
                d.get("outside_ms", 0) / n, d.get("sync_ms", 0) / n, (d.get("fill_ms", 0) - d.get("sync_ms", 0)) / n)
    self._last = dict(self.stats)


NvTier.log_stats = _log_stats


# ---- d1b4: tier-mode staged prefill (NVMe -> pinned USM host -> copy engine -> VRAM staging, double buffered)
# The prefill kernel reads VRAM instead of first-touching freshly filled SVM pages (~1.6 GB/s measured, srv11 prof).
# For layer L+1 the reader pool starts O_DIRECT reads of every non-VRAM-resident expert as soon as layer L's kernel is
# queued; the CPU runs ahead of the GPU, so the reads overlap layer L's compute. The H2D copy runs on a side stream.

class _AnonBuf:
    """2 MiB-aligned anonymous THP buffer (O_DIRECT target and copy-engine source)."""
    def __init__(self, size):
        self.m = mmap.mmap(-1, size + MiB2)
        p = ctypes.addressof(ctypes.c_char.from_buffer(self.m)); self.p = (p + MiB2 - 1) // MiB2 * MiB2
        libc.madvise(ctypes.c_void_p(self.p), ctypes.c_size_t(size), 14)
    def data_ptr(self): return self.p


def _stage_setup(self):
    from concurrent.futures import ThreadPoolExecutor
    E, rec = self.E, self.rec
    NB = int(os.environ.get("EXL3_NVTIER_STAGE_HOSTBUF", "4"))           # host bounce ring (read-ahead NB-1 layers)
    sg = dict(dev=[torch.empty(E * rec, dtype=torch.uint8, device=self.dev) for _ in range(2)],
              host=[_AnonBuf(E * rec) for _ in range(NB)],     # O_DIRECT cannot target USM host (EFAULT); the copy
                                                               # engine reads anonymous THP memory at 26.7 GB/s (h2d_test)
              NB=NB, cs=torch.xpu.Stream(), comp_ev=[None, None], hfree={}, pending={}, nexth=0,
              pool=ThreadPoolExecutor(int(os.environ.get("EXL3_NVTIER_FILL_THREADS", "32"))),
              order=[int(li) for li in np.argsort(self.ckpt, kind="stable")], snap=None, snap_dev=None, dbuf=0,
              ar=torch.arange(E, dtype=torch.int64, device=self.dev) * rec,
              prof=dict(layers=0, wait_ms=0.0, bytes=0, t0=None))
    sg["pos"] = {li: i for i, li in enumerate(sg["order"])}
    self._sg = sg


def _stage_kick(self, li, _buf=None):
    """Start O_DIRECT reads of layer li's non-VRAM experts into the next host ring buffer (reader threads)."""
    sg = self._sg; E, rec = self.E, self.rec
    hb = sg["nexth"]; sg["nexth"] = (hb + 1) % sg["NB"]
    ev = sg["hfree"].pop(hb, None)
    if ev is not None:
        ev.synchronize()                          # the H2D that last read this host buffer is done (NB-1 layers ago)
    miss = np.nonzero(sg["snap"][li * E:(li + 1) * E] < 0)[0]
    hp = sg["host"][hb].data_ptr(); nfd = self.nfd
    def rd(e):
        b = (ctypes.c_char * rec).from_address(hp + int(e) * rec)
        n = os.preadv(nfd, [b], self.rec_index(li * E + int(e)) * rec); assert n == rec, (n, rec)
    sg["pending"][li] = (hb, miss, [sg["pool"].submit(rd, e) for e in miss.tolist()])


def _stage_forward(self, store, key, li, x, ids, w):
    if not hasattr(self, "_sg"): self._stage_setup()
    sg = self._sg; E, rec = self.E, self.rec
    pos = sg["pos"][li]; order = sg["order"]; D = sg["NB"] - 1
    if pos == 0 or li not in sg["pending"]:
        # first MoE layer of a prefill forward: one sync per forward - VRAM cache may have moved (decode graph replays)
        torch.xpu.synchronize()
        sg["snap_dev"] = store.slot_of_dev[: self.L * E].to(torch.int64).clone()
        sg["snap"] = sg["snap_dev"].cpu().numpy()
        for _, _, fs in sg["pending"].values():
            for f in fs: f.result()
        sg["pending"].clear()
        sg["prof"]["t0"] = time.perf_counter()
        for q in range(pos, min(pos + 1 + D, len(order))):
            self._stage_kick(order[q])
    hb, miss, futs = sg["pending"].pop(li)
    t = time.perf_counter()
    for f in futs: f.result()
    sg["prof"]["wait_ms"] += (time.perf_counter() - t) * 1e3
    TP = os.environ.get("EXL3_NVTIER_STAGE_PROF", "0") == "1"
    mk = (lambda: torch.xpu.Event(enable_timing=True)) if TP else None
    buf = sg["dbuf"]; sg["dbuf"] = 1 - buf
    cs, cur = sg["cs"], torch.xpu.current_stream()
    dbase, hbase = sg["dev"][buf].data_ptr(), sg["host"][hb].data_ptr()
    with torch.xpu.stream(cs):
        if sg["comp_ev"][buf] is not None:
            cs.wait_event(sg["comp_ev"][buf])     # the kernel that last read this VRAM buffer is done (device-side)
        if TP: h0 = mk(); h0.record(cs)
        i, m = 0, miss.tolist()
        while i < len(m):
            j = i
            while j + 1 < len(m) and m[j + 1] == m[j] + 1: j += 1
            store.X.memcpy_async(s64(dbase + m[i] * rec), s64(hbase + m[i] * rec), (j - i + 1) * rec)
            i = j + 1
        ev = torch.xpu.Event(enable_timing=TP); ev.record(cs)
    sg["hfree"][hb] = ev
    if TP: c0 = mk(); c0.record(cur)
    cur.wait_event(ev)
    if TP: c1 = mk(); c1.record(cur)
    so = sg["snap_dev"][li * E:(li + 1) * E]                   # device-side table: no host round trip per layer
    table = torch.where(so >= 0, store.slot_base + so.clamp_min(0) * store.blob, dbase + sg["ar"])
    y = store.forward(key, x, ids, w, ptrs=table)
    ce = torch.xpu.Event(enable_timing=TP); ce.record(cur); sg["comp_ev"][buf] = ce
    if TP: sg.setdefault("tev", []).append((h0, ev, c0, c1, ce, len(miss)))
    sg["prof"]["layers"] += 1; sg["prof"]["bytes"] += len(miss) * rec
    nq = pos + 1 + D
    if nq < len(order) and order[nq] not in sg["pending"]:
        self._stage_kick(order[nq])
    if pos + 1 == len(order) and TP and sg.get("tev"):
        torch.xpu.synchronize()
        T = sg.pop("tev"); el = lambda a, b: a.elapsed_time(b)
        h2d = sum(el(h0, ev) for h0, ev, *_ in T); wait = sum(el(c0, c1) for _, _, c0, c1, _, _ in T)
        kern = sum(el(c1, ce) for *_, c1, ce, _ in T); wall = el(T[0][2], T[-1][4])
        gaps = [(el(T[i][4], T[i + 1][2]), int(self.ckpt[order[i + 1]])) for i in range(len(T) - 1)]   # MoE end -> next MoE start
        gs = sorted(g for g, _ in gaps)
        import logging
        logging.getLogger("exl3xpu").warning("nvtier stage gaps (non-MoE GPU time between MoE layers): median %.0f ms, p90 %.0f ms, top %s",
            gs[len(gs) // 2], gs[int(0.9 * len(gs))], ", ".join(f"L{l}:{g:.0f}ms" for g, l in sorted(gaps, reverse=True)[:5]))
        logging.getLogger("exl3xpu").warning("nvtier stage GPU prof: rows %d layers %d | GPU wall %.0f ms = MoE kernels %.0f + waiting on H2D %.0f + other (attn/GDN/dense/launch gaps) %.0f | H2D busy %.0f ms (%.1f GB/s) | host reader-wait %.0f ms (read-ahead %d)",
            x.shape[0], len(T), wall, kern, wait, wall - kern - wait, h2d, sum(t[5] for t in T) * rec / 1e6 / max(h2d, 1e-3), sg["prof"]["wait_ms"], D)
        sg["prof"].update(layers=0, wait_ms=0.0, bytes=0)
    return y


NvTier._stage_setup = _stage_setup
NvTier._stage_kick = _stage_kick
NvTier.stage_forward = _stage_forward


# ---- N130: decode-fill queue depth (EXL3_NVTIER_DECODE_FILL_QUEUES=N, default 1 = unchanged single worker).
# N workers, each with its own FIFO; a key's fill and punch always go to the same worker (key % N), so the
# per-key order (punch queued before a later re-fill) is the same as with one worker. Up to N O_DIRECT reads in flight.
import queue as _queue, threading as _threading


class _RouterQ:
    def __init__(self, qs): self.qs = qs
    def get(self): return self.qs[0].get()          # the original NvTierFast._worker reads self.q.get() -> queue 0
    def empty(self): return all(q.empty() for q in self.qs)
    def put(self, item):
        op, key = item
        if op == "stop":
            for q in self.qs: q.put(item)
        else:
            self.qs[int(key) % len(self.qs)].put(item)


def _worker_q(self, q):
    buf = mmap.mmap(-1, self.rec); bufp = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    while True:
        op, key = q.get()
        if op == "fill":
            t = time.perf_counter()
            n = os.preadv(self.nfd, [buf], self.rec_index(key) * self.rec); assert n == self.rec
            ctypes.memmove(self.st.addr(key), bufp, self.blob)
            self.done.put(key)
            self.stats["nvme_reads"] += 1; self.stats["nvme_ms"] += (time.perf_counter() - t) * 1e3
        elif op == "punch":
            if not getattr(self, "no_punch", False) and not self.res[key]: self._punch(key)
        elif op == "stop":
            return


_fast_init = NvTierFast.__init__


def _fast_init_n130(self, *a, **k):
    _fast_init(self, *a, **k)
    nq = int(os.environ.get("EXL3_NVTIER_DECODE_FILL_QUEUES", "1"))
    if nq > 1:
        q0 = self.q                                  # worker 0 = the original thread, keeps consuming q0
        qs = [q0] + [_queue.Queue() for _ in range(nq - 1)]
        self._ths = [_threading.Thread(target=_worker_q, args=(self, q), daemon=True) for q in qs[1:]]
        for th in self._ths: th.start()
        self.q = _RouterQ(qs)
        import logging
        logging.getLogger("exl3xpu").warning("nvtier N130: %d decode-fill queues", nq)


NvTierFast.__init__ = _fast_init_n130


# ---- N130: admission-controlled decode tier (EXL3_NVTIER_ADMIT=1; unset = the srv23 NvTierAsync.step unchanged).
# Failure it fixes (N110 attempt 1, N130 s16b): when the cold decode working set exceeds the RAM budget, queued fills
# (inflight) reserve the whole budget, every completed fill is the oldest non-VRAM key and is evicted before the GPU
# reads it, punches queue behind the fill backlog (memfd grows past the budget), and the tier collapses to
# ~0.1 GB resident with ~470 of 480 picks masked per step (garbage / runaway text).
# Policy: (1) evict RAM copies of VRAM-resident experts first (as before); (2) evict other keys only once they are at
# least ADMIT_AGE steps old (a fresh fill gets time to be read and promoted to VRAM); (3) admit new fills only into
# free budget, at most ADMIT_QMAX in flight, most-missed keys first; the rest stay masked and are re-logged on their
# next miss.
_async_step_orig = NvTierAsync.step


def _async_step_admit(self):
    if os.environ.get("EXL3_NVTIER_ADMIT", "0") != "1":
        return _async_step_orig(self)
    AGE = int(os.environ.get("EXL3_NVTIER_ADMIT_AGE", "4")); QMAX = int(os.environ.get("EXL3_NVTIER_ADMIT_QMAX", "256"))
    t0 = time.perf_counter()
    self.tick += 1
    if getattr(self, "_t_end", None) is not None:
        self.stats["outside_ms"] = self.stats.get("outside_ms", 0.0) + (t0 - self._t_end) * 1e3
    n = self._n
    cur = torch.xpu.current_stream()
    k = self._k; self._k ^= 1
    self._h_res[k].copy_(self.res_dev[:n], non_blocking=True)
    self._h_ring[k].copy_(self.ring, non_blocking=True)
    self._h_slot[k].copy_(self.st.slot_of_dev[:n], non_blocking=True)
    ev = torch.xpu.Event(); ev.record(cur); self._snap_ev[k] = ev
    j = k ^ 1
    if self._snap_ev[j] is None:
        self._t_end = time.perf_counter(); self.stats["steps"] += 1
        return 0
    t_s = time.perf_counter()
    self._snap_ev[j].synchronize()
    self.stats["sync_ms"] = self.stats.get("sync_ms", 0.0) + (time.perf_counter() - t_s) * 1e3
    res_d = self._h_res[j].numpy().copy()
    if self._last_diff is not None:
        dk, dv = self._last_diff
        res_d[dk] = dv
    ring = self._h_ring[j].numpy()
    vram = self._h_slot[j].numpy() >= 0
    keep = []
    for ce, keys in self._clears:
        if ce.query():
            for kk in keys.tolist():
                if res_d[kk]:
                    self.stats["evict_rescued"] = self.stats.get("evict_rescued", 0) + 1
                else:
                    self.q.put(("punch", kk))
        else:
            keep.append((ce, keys))
    self._clears = keep
    newly = (res_d == 1) & (self.res == 0)
    self.age[newly] = self.tick
    if hasattr(self, "transient"): self.transient[newly] = False
    res = self.res.copy()
    res[newly] = 1
    while not self.done.empty():
        kk = self.done.get(); res[kk] = 1; self.age[kk] = self.tick; self.inflight[kk] = False
    # new misses (with multiplicity -> priority)
    cnt = int(ring[0]); newm = cnt - self.ring_seen
    cap = self.ring.numel() - 1
    if newm > cap: self.stats["ring_overflow"] += newm - cap; newm = cap
    want = np.zeros(0, dtype=np.int64)
    if newm > 0:
        lo = (cnt - newm) % cap; hi = cnt % cap
        r = ring[1:]
        keys = np.concatenate([r[lo:], r[:hi]]) if hi <= lo else r[lo:hi]
        u, c = np.unique(keys[:newm].astype(np.int64), return_counts=True)
        ok = (res[u] == 0) & (~self.inflight[u])
        u, c = u[ok], c[ok]
        want = u[np.argsort(-c, kind="stable")]
    self.ring_seen = cnt; self.stats["masked"] += max(0, newm)
    # budget: VRAM duplicates first, then non-VRAM keys older than AGE steps
    inflight_n = int(self.inflight.sum())
    over = int(res.sum()) + inflight_n + len(want) - self.budget
    ev_keys = np.zeros(0, dtype=np.int64)
    if over > 0:
        cand1 = np.nonzero((res == 1) & vram & ~self.pin_mask)[0]
        take1 = cand1[np.argsort(self.age[cand1])[:over]]
        over -= len(take1)
        take2 = np.zeros(0, dtype=np.int64)
        if over > 0:
            m = (res == 1) & ~self.pin_mask & (self.age <= self.tick - AGE); m[take1] = False
            cand2 = np.nonzero(m)[0]
            take2 = cand2[np.argsort(self.age[cand2])[:over]]
        ev_keys = np.concatenate([take1, take2]).astype(np.int64)
        res[ev_keys] = 0
        self.stats["evicts"] += len(ev_keys)
    free = min(self.budget - int(res.sum()) - inflight_n, QMAX - inflight_n)
    adm = want[:max(0, free)]
    self.stats["admit_dropped"] = self.stats.get("admit_dropped", 0) + (len(want) - len(adm))
    for kk in adm.tolist():
        self.inflight[kk] = True; self.q.put(("fill", kk))
    diff = np.nonzero(res != res_d)[0].astype(np.int64)
    if len(diff):
        idx = torch.from_numpy(diff).pin_memory().to(self.dev, non_blocking=True)
        val = torch.from_numpy(res[diff].copy()).pin_memory().to(self.dev, non_blocking=True)
        self.res_dev.index_copy_(0, idx, val)
        self._last_diff = (diff, res[diff].copy())
    else:
        self._last_diff = None
    if len(ev_keys):
        ce = torch.xpu.Event(); ce.record(cur); self._clears.append((ce, ev_keys))
    self.res = res
    self.stats["steps"] += 1
    self._t_end = time.perf_counter()
    self.stats["fill_ms"] += (self._t_end - t0) * 1e3
    return int(max(0, newm))


NvTierAsync.step = _async_step_admit


_log_stats_orig = NvTier.log_stats


def _log_stats_n130(self, logger, every=200):
    if not (self.stats["steps"] % every or not self.stats["steps"]):
        import logging
        logging.getLogger("exl3xpu").info("exl3xpu: nvtier N130 inflight %d admit_dropped(total) %d memfd %.2f GB",
                                          int(self.inflight.sum()) if hasattr(self, "inflight") else -1,
                                          self.stats.get("admit_dropped", 0), os.fstat(self.st.fd).st_blocks * 512 / 2 ** 30)
    return _log_stats_orig(self, logger, every)


NvTier.log_stats = _log_stats_n130

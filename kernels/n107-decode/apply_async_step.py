"""n107-decode: async NvTierFast.step (no full device sync per decode token).

Apply on omarchy:  python3 apply_async_step.py ~/freetoken-exl3/runs/N104-nvtier
 - appends class NvTierAsync to src/exl3xpu/nvtier.py (backup nvtier.py.pre_n107)
 - makes sglang_plugin.tier.py pick NvTierAsync when EXL3_NVTIER_ASYNC=1 (backup .pre_n107)
 - n104_serve.sh passes -e EXL3_NVTIER_ASYNC=${NVASYNC:-0}

Stream order per decode step N (all on the current stream):
  replay_N  ->  step(): snapshot_N (async D2H of res_dev, ring, slot_of_dev into pinned buffers), event e_N
                       consume snapshot_{N-1} (e_{N-1} is done or nearly: it was queued before replay_N)
                       diff_N (async H2D flag updates), clear-event c_N for evicted keys
Snapshot_{N-1} predates diff_{N-1}, so before comparing it with the host state we overlay diff_{N-1} onto it
(device state after diff_{N-1} == snapshot_{N-1} with those keys overwritten). Evicted keys are punched only once
their clear event has completed: every kernel queued before the flag clear (the only ones that may read the page)
has finished by then. Fills stay masked one step longer than in NvTierFast (two steps instead of one).
"""
import os, re, shutil, sys

CLS = r'''


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
'''

def main(root):
    nv = os.path.join(root, "src/exl3xpu/nvtier.py")
    pl = os.path.join(root, "sglang_plugin.tier.py")
    sv = os.path.join(root, "n104_serve.sh")
    s = open(nv).read()
    if "class NvTierAsync" not in s:
        shutil.copy(nv, nv + ".pre_n107")
        # insert before the module-level helpers that follow NvTierFast (first top-level def after the class)
        anchor = "\n\ndef _ensure_for_prefill("
        assert anchor in s, "anchor _ensure_for_prefill not found"
        s = s.replace(anchor, CLS + anchor, 1)
        open(nv, "w").write(s)
    p = open(pl).read()
    if "EXL3_NVTIER_ASYNC" not in p:
        shutil.copy(pl, pl + ".pre_n107")
        m = re.search(r"^([ \t]*)_NVTIER = NvTierFast\(", p, re.M)
        assert m, "plugin NvTierFast construction not found"
        ind = m.group(1)
        p = p[:m.start()] + (ind + "_cls = NvTierAsync if os.environ.get(\"EXL3_NVTIER_ASYNC\", \"0\") == \"1\" else NvTierFast\n"
                             + ind + "_NVTIER = _cls(") + p[m.end():]
        # import NvTierAsync wherever NvTierFast is imported from the nvtier module
        p, k = re.subn(r"(from [\w.]*nvtier import [^\n]*\bNvTierFast\b)(?![^\n]*NvTierAsync)", r"\1, NvTierAsync", p)
        assert k >= 1, "nvtier import line not found"
        open(pl, "w").write(p)
    v = open(sv).read()
    if "EXL3_NVTIER_ASYNC" not in v:
        v = v.replace("-e EXL3_NVTIER_STAGE=${STAGE:-0}", "-e EXL3_NVTIER_STAGE=${STAGE:-0} -e EXL3_NVTIER_ASYNC=${NVASYNC:-0}", 1)
        open(sv, "w").write(v)
    print("applied; check: grep -n NvTierAsync", nv, pl)

if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/freetoken-exl3/runs/N104-nvtier"))

"""
Two-tier EXL3 routed-expert store for the B70 (XPU): every expert of every MoE layer lives in USM host memory
(tier 1, read zero-copy over PCIe by the kernels), a subset lives in device slots (tier 0, the expert cache). The
grouped MoE kernels (`torch.ops.exl3xpu_moe.moe_forward`, csrc/exl3_moe.sycl) address experts only through a
per-layer POINTER TABLE, so moving an expert between tiers is a single 8-byte table write; both pointers are always
valid, so a table update never races a kernel into reading garbage.

Expert blob (one per expert, `blob_bytes(H, I, K)` bytes, 1,862,400 B for H=2560, I=640, K=3):
  gate|up trellis (u32, [H/16][2I/16][8K], PLANAR4 word order) | down trellis ([I/16][H/16][8K], PLANAR4) |
  suh_g[H] suh_u[H] svh_g[I] svh_u[I] suh_d[I] svh_d[H]   (fp16)
PLANAR4: in every k-row the tiles are grouped by 4 along n; a group's 4*8K words are stored plane-major
([plane i][tile][period g]) instead of tile-major ([tile][3g + i]) -- see pack_expert().

API (all device work is enqueued on the current XPU stream unless noted; nothing syncs the host):
  store = ExpertStore(H, I, K, n_experts, n_slots)          # n_slots device slots shared by all layers
  store.add_layer(key, blobs_cpu [E, BLOB] uint8 | None)     # allocates the layer's host USM arena (fills it)
  store.host_view(key)[e]                                    # uint8 view of expert e's host blob
  store.ptrs(key)                                            # int64 [E] device pointer table (fixed address)
  store.make_resident(key, experts, stream=None)             # copy blobs into free/evicted slots, then repoint
  store.evict(key, experts)                                  # repoint to host (slot freed; reuse is stream-ordered)
  store.forward(key, x, topk_ids, topk_w) -> out             # routed-MoE output (weights applied, shared expert NOT)
  store.forward_cached(key, x, topk_ids, topk_w)            # decode with the DEVICE-managed LRU cache: misses are read
                                                             #   zero-copy and written through into LRU slots, the
                                                             #   table is repointed by the same call (no host sync)
  store.prefetch(key, predicted_ids)                         # on a side stream between MoE calls: claim LRU slots for
                                                             #   predicted experts, gather-copy them, repoint (X006)
  store.stage_layer(key, buf, stream) -> ptrs                # prefill: copy the layer's non-resident experts into
                                                             #   staging buffer buf (0/1) on `stream`; returns a
                                                             #   pointer table (slots for residents, staging else)
  store.slot_of[key]  (cpu int32 [E], -1 = host only)        # the slot map
"""
from __future__ import annotations

import os
import threading

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_lib_loaded = False
_lock = threading.Lock()


def ops():
    """torch.ops.exl3xpu_moe, loading exl3xpu/_moe.so (or $EXL3_MOE_LIB) once."""
    global _lib_loaded
    with _lock:
        if not _lib_loaded:
            torch.ops.load_library(os.environ.get("EXL3_MOE_LIB") or os.path.join(_HERE, "_moe.so"))

            @torch.library.register_fake("exl3xpu_moe::moe_forward")
            def _fake(x, topk_ids, topk_w, ptrs, I, K, n_experts):   # noqa: N803
                return torch.empty_like(x)

            if hasattr(torch.ops.exl3xpu_moe, "moe_forward_cached"):
                @torch.library.register_fake("exl3xpu_moe::moe_forward_cached")
                def _fake_c(x, *args):
                    return torch.empty_like(x)

            _lib_loaded = True
    return torch.ops.exl3xpu_moe


def s64(p: int) -> int:
    return p - (1 << 64) if p >= (1 << 63) else p


def planar4(tr: torch.Tensor, K: int) -> torch.Tensor:
    """int16 trellis [rows, tiles, 16K] -> int32 words in PLANAR4 order (K with gcd(K, 32) = 1: 8 periods/tile)."""
    assert K in (3, 5, 7), "PLANAR4 packing implemented for odd K (D = K words per 32-value period)"
    r, n, _ = tr.shape
    w = tr.contiguous().view(torch.int32).view(r, n // 4, 4, 8, K)       # [row, group, tile, period g, plane i]
    return w.permute(0, 1, 4, 2, 3).contiguous()                         # [row, group, plane i, tile, g]


def pack_expert(gate: dict, up: dict, down: dict, K: int, out: torch.Tensor | None = None) -> torch.Tensor:
    """Checkpoint tensors {trellis, suh, svh} of one expert -> blob (uint8 CPU tensor, or written into `out`)."""
    b = lambda z: z.contiguous().view(torch.uint8).flatten()     # noqa: E731
    gu = torch.cat([gate["trellis"], up["trellis"]], dim=1)
    parts = [b(planar4(gu, K)), b(planar4(down["trellis"], K)), b(gate["suh"].half()), b(up["suh"].half()),
             b(gate["svh"].half()), b(up["svh"].half()), b(down["suh"].half()), b(down["svh"].half())]
    if out is None:
        return torch.cat(parts)
    o = 0
    for p in parts:
        out[o:o + p.numel()].copy_(p)
        o += p.numel()
    assert o == out.numel(), (o, out.numel())
    return out


class ExpertStore:
    def __init__(self, H: int, I: int, K: int, n_experts: int, n_slots: int, device=None, max_layers: int = 64):
        self.H, self.I, self.K, self.E = H, I, K, n_experts
        self.dev = device or torch.device("xpu", torch.xpu.current_device())
        self.X = ops()
        self.blob = int(self.X.blob_bytes(H, I, K))
        self.n_slots = n_slots
        self.max_layers = max_layers
        self.slots = torch.empty((n_slots, self.blob), dtype=torch.uint8, device=self.dev) if n_slots else None
        self.slot_owner: list = [None] * n_slots            # slot -> (key, e)   (static placement mirror)
        self.free_slots = list(range(n_slots - 1, -1, -1))
        self.host: dict = {}                                # key -> uint8 CPU tensor [E, BLOB] (USM host)
        self.layer_index: dict = {}                         # key -> row in the device tables
        self.slot_of: dict = {}                             # key -> cpu int32 [E] (mirror of static placement)
        self._host_ptr: dict = {}
        self._stage = [None, None]
        self._slot_ready_ev: dict = {}
        # device-side cache state (authoritative for forward_cached)
        LE = max_layers * n_experts
        self.ptrs_all = torch.zeros(LE, dtype=torch.int64, device=self.dev)
        self.slot_of_dev = torch.full((LE,), -1, dtype=torch.int32, device=self.dev)
        self.slot_key = torch.full((n_slots,), -1, dtype=torch.int32, device=self.dev)
        self.slot_last = torch.full((n_slots,), -1, dtype=torch.int32, device=self.dev)
        self.tick = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.host_base = torch.zeros(max_layers, dtype=torch.int64, device=self.dev)
        self.fill_all = torch.zeros(LE, dtype=torch.int64, device=self.dev)
        self.fill_list = torch.zeros(1 + n_experts, dtype=torch.int32, device=self.dev)
        self.pf_list = torch.zeros(1 + n_experts, dtype=torch.int32, device=self.dev)
        self.pf_slot = torch.zeros(n_experts, dtype=torch.int32, device=self.dev)
        self.pend = torch.zeros(max_layers * 35, dtype=torch.int32, device=self.dev)      # copy-engine prefetch state
        self.done_seq = torch.zeros(0, dtype=torch.int32, device=self.dev)               # set by start_copy_prefetch
        self.slot_base = s64(self.slots.data_ptr()) if n_slots else 0
        self._cache_used = False            # set by forward_cached: static ops must resync from the device first

    # ---- layers
    def add_layer(self, key, blobs: torch.Tensor | None = None) -> torch.Tensor:
        li = len(self.layer_index)
        if li >= self.max_layers:
            raise RuntimeError(f"ExpertStore: more than max_layers={self.max_layers} MoE layers")
        self.layer_index[key] = li
        h = self.X.host_alloc(self.E * self.blob).view(self.E, self.blob)
        if blobs is not None:
            h.copy_(blobs)
        self.host[key] = h
        base = s64(h.data_ptr())
        self._host_ptr[key] = base + torch.arange(self.E, dtype=torch.int64) * self.blob
        self.ptrs_all[li * self.E:(li + 1) * self.E].copy_(self._host_ptr[key])
        self.host_base[li] = base
        self.slot_of[key] = torch.full((self.E,), -1, dtype=torch.int32)
        return h

    def host_view(self, key) -> torch.Tensor:
        return self.host[key]

    def ptrs(self, key) -> torch.Tensor:
        li = self.layer_index[key]
        return self.ptrs_all[li * self.E:(li + 1) * self.E]

    def slot_ptr(self, s: int) -> int:
        return self.slot_base + s * self.blob

    # ---- static residency (load time / host-driven policies)
    def make_resident(self, key, experts, stream=None) -> int:
        """Copy experts into free device slots, then repoint the table (and the device slot map). Copies and the
        table update are ordered on `stream` (default: current stream). Returns #copied."""
        s_ = stream or torch.xpu.current_stream()
        self._resync()
        so = self.slot_of[key]
        li = self.layer_index[key]
        todo = [int(e) for e in experts if so[int(e)] < 0]
        if not todo:
            return 0
        es, sls = [], []
        with torch.xpu.stream(s_):
            for e in todo:
                if not self.free_slots:
                    raise RuntimeError("ExpertStore: no free slot (evict first or raise n_slots)")
                sl = self.free_slots.pop()
                ev = self._slot_ready_ev.pop(sl, None)
                if ev is not None:
                    s_.wait_event(ev)
                self.X.memcpy_async(self.slot_ptr(sl), s64(self.host[key][e].data_ptr()), self.blob)
                self.slot_owner[sl] = (key, e)
                so[e] = sl
                es.append(e)
                sls.append(sl)
            keys = torch.tensor([li * self.E + e for e in es], dtype=torch.int64)
            slt = torch.tensor(sls, dtype=torch.int64)
            d = self.dev
            self.ptrs_all.index_copy_(0, keys.to(d), (self.slot_base + slt * self.blob).to(d))
            self.slot_of_dev.index_copy_(0, keys.to(d), slt.to(torch.int32).to(d))
            self.slot_key.index_copy_(0, slt.to(d), keys.to(torch.int32).to(d))
            self.slot_last.index_fill_(0, slt.to(d), 0)
        return len(todo)

    def evict(self, key, experts) -> None:
        """Repoint to the host copy; the slot becomes reusable after the current stream's already-queued work."""
        self._resync()
        so = self.slot_of[key]
        li = self.layer_index[key]
        es, sls = [], []
        for e in experts:
            e = int(e)
            sl = int(so[e])
            if sl < 0:
                continue
            es.append(e)
            sls.append(sl)
            so[e] = -1
            self.slot_owner[sl] = None
        if not es:
            return
        d = self.dev
        et = torch.tensor(es, dtype=torch.int64)
        keys = (li * self.E + et).to(d)
        slt = torch.tensor(sls, dtype=torch.int64).to(d)
        self.ptrs_all.index_copy_(0, keys, self._host_ptr[key][et].to(d))
        self.slot_of_dev.index_fill_(0, keys, -1)
        self.slot_key.index_fill_(0, slt, -1)
        self.slot_last.index_fill_(0, slt, -1)
        ev = torch.xpu.Event()
        ev.record(torch.xpu.current_stream())
        for sl in sls:
            self._slot_ready_ev[sl] = ev
            self.free_slots.append(sl)

    def resident_count(self, key=None) -> int:
        if key is None:
            return sum(int((v >= 0).sum()) for v in self.slot_of.values())
        return int((self.slot_of[key] >= 0).sum())

    def device_resident_count(self, key) -> int:
        """From the device slot map (syncs; debug/metrics only)."""
        li = self.layer_index[key]
        return int((self.slot_of_dev[li * self.E:(li + 1) * self.E] >= 0).sum().item())

    # ---- compute
    @staticmethod
    def _rt(topk_ids, topk_w):
        ids = topk_ids if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous() else topk_ids.to(torch.int32).contiguous()
        w = topk_w if topk_w.dtype == torch.float32 and topk_w.is_contiguous() else topk_w.to(torch.float32).contiguous()
        return ids, w

    def forward(self, key, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor, ptrs=None) -> torch.Tensor:
        ids, w = self._rt(topk_ids, topk_w)
        return self.X.moe_forward(x, ids, w, self.ptrs(key) if ptrs is None else ptrs, self.I, self.K, self.E)

    def forward_cached(self, key, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                       max_fill: int | None = None) -> torch.Tensor:
        """Decode with the device-managed LRU expert cache: hits read their slot, misses read host memory zero-copy
        and are written through into the LRU slot (committed at the end of the call). No host sync, graph-safe.
        Do not mix with make_resident/evict on the same slots while cached calls are queued."""
        ids, w = self._rt(topk_ids, topk_w)
        self._cache_used = True
        return self.X.moe_forward_cached(x, ids, w, self.ptrs_all, self.layer_index[key], self.slot_of_dev,
                                         self.slot_key, self.slot_last, self.tick, self.host_base, self.slot_base,
                                         self.fill_all, self.fill_list, self.E if max_fill is None else max_fill,
                                         self.I, self.K, self.E, self.pend, self.done_seq)

    def prefetch(self, key, ids: torch.Tensor, max_fill: int = 16) -> None:
        """Claim LRU slots for the (predicted) experts `ids` of layer `key` and copy them from host memory, on the
        CURRENT stream -- use a side stream that waited for the previous MoE call, and make the compute stream wait
        for it before the next MoE call. The tick is not advanced, so slots of the last MoE call are protected."""
        if not self.n_slots:
            return
        ids = ids if ids.dtype == torch.int32 and ids.is_contiguous() else ids.to(torch.int32).contiguous()
        self._cache_used = True
        self.X.moe_prefetch(ids.flatten(), self.ptrs_all, self.layer_index[key], self.slot_of_dev, self.slot_key,
                            self.slot_last, self.tick, self.host_base, self.slot_base, self.pf_list, self.pf_slot,
                            max_fill, self.E, self.blob)

    # ---- copy-engine prefetch (device plan -> host worker -> queue.memcpy), X006b
    def start_copy_prefetch(self) -> None:
        """Start the C++ worker that turns device-written prefetch plans into copy-engine transfers."""
        L, E = len(self.layer_index), self.E
        self._plans = self.X.host_alloc(L * (4 + 2 * E) * 4)
        self._plans.zero_()
        self._plan_seq = torch.zeros(L, dtype=torch.int32, device=self.dev)
        self.done_seq = torch.zeros(L, dtype=torch.int32, device=self.dev)
        self.pend.zero_()
        self._commit_host = self.X.host_alloc(L * 2 * E * 12)
        hb = torch.zeros(L, dtype=torch.int64)
        for key, li in self.layer_index.items():
            hb[li] = int(self._host_ptr[key][0])
        torch.xpu.synchronize()
        self._pf_stream = torch.xpu.Stream()
        with torch.xpu.stream(self._pf_stream):     # the worker submits its copies to this stream's queue
            self.X.moe_prefetch_worker_start(self._plans, hb, self.slot_base, self.blob, self.ptrs_all, self.slot_last,
                                             self._commit_host, self.done_seq, L, E)

    def plan_prefetch(self, key, ids: torch.Tensor, max_fill: int = 16) -> None:
        """On the COMPUTE stream (e.g. right after MoE(l) for layer l+1): pin LRU slots for the predicted experts and
        publish the plan; the worker copies them on the copy engine and commits. The compute stream never waits."""
        ids = ids if ids.dtype == torch.int32 and ids.is_contiguous() else ids.to(torch.int32).contiguous()
        self._cache_used = True
        self.X.moe_prefetch_plan(ids.flatten(), self.ptrs_all, self.layer_index[key], self.slot_of_dev, self.slot_key,
                                 self.slot_last, self.tick, self.host_base, self.slot_base, self.pf_list, self.pf_slot,
                                 self._plans, self._plan_seq, self.pend, self.done_seq, max_fill, self.E, self.blob)

    def stop_copy_prefetch(self):
        """Stop the worker (drains its queue), then commit what finished (one empty ensure pass); returns
        [plans handled, experts copied]. Afterwards forward_cached runs without prefetch commits."""
        r = self.X.moe_prefetch_worker_stop()
        if self.done_seq.numel():
            k0 = next(iter(self.layer_index))
            x = torch.zeros((1, self.H), dtype=torch.bfloat16, device=self.dev)
            ids = torch.zeros((1, 1), dtype=torch.int32, device=self.dev)
            self.forward_cached(k0, x, ids, torch.ones((1, 1), device=self.dev), max_fill=0)   # commit pass
            torch.xpu.synchronize()
            self.done_seq = torch.zeros(0, dtype=torch.int32, device=self.dev)
        return r

    # ---- prefill staging (FreeToken-style layer streaming)
    def stage_layer(self, key, buf: int, stream) -> torch.Tensor:
        """Copy every non-resident expert of `key` into staging buffer `buf` on `stream`; returns the pointer table
        to pass to forward(ptrs=...) once `stream` has been waited on. Contiguous expert runs are copied together.
        Uses the static-placement mirror (slot_of); after forward_cached calls, refresh it with sync_mirror()."""
        if self._stage[buf] is None:
            self._stage[buf] = torch.empty((self.E, self.blob), dtype=torch.uint8, device=self.dev)
        st = self._stage[buf]
        self._resync()
        so = self.slot_of[key]
        miss = (so < 0).nonzero().flatten().tolist()
        with torch.xpu.stream(stream):
            i = 0
            while i < len(miss):
                j = i
                while j + 1 < len(miss) and miss[j + 1] == miss[j] + 1:
                    j += 1
                n = j - i + 1
                self.X.memcpy_async(s64(st[miss[i]].data_ptr()), s64(self.host[key][miss[i]].data_ptr()), n * self.blob)
                i = j + 1
        sp = s64(st.data_ptr()) + torch.arange(self.E, dtype=torch.int64) * self.blob
        if self.n_slots:
            dp = self.slot_base + so.clamp_min(0).to(torch.int64) * self.blob
            sp = torch.where(so >= 0, dp, sp)
        # blocking H2D: a non_blocking copy from this pageable temporary may run after it is freed (XPU)
        return sp.to(self.dev)

    def reset_cache(self) -> None:
        """Empty every slot and repoint all tables to host memory (current stream; waits for queued work)."""
        torch.xpu.current_stream().synchronize()
        for key, li in self.layer_index.items():
            self.ptrs_all[li * self.E:(li + 1) * self.E].copy_(self._host_ptr[key])
            self.slot_of[key].fill_(-1)
        self.slot_of_dev.fill_(-1)
        self.slot_key.fill_(-1)
        self.slot_last.fill_(-1)
        self.fill_all.zero_()
        self.pend.zero_()
        self.slot_owner = [None] * self.n_slots
        self.free_slots = list(range(self.n_slots - 1, -1, -1))
        self._slot_ready_ev.clear()
        self._cache_used = False

    # ---- prefill layer streaming inside a model forward (copy engine, double buffer), X004
    def forward_prefill_staged(self, key, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
        """Prefill-size MoE for layer `key` from a device staging buffer: the layer's non-resident experts are copied
        H2D on a side stream (copy engine) -- normally already done while the previous layer computed -- then this call
        starts staging the NEXT layer into the other buffer. Resident experts are read from their cache slots."""
        if not hasattr(self, "_pf"):
            self._pf = {"stream": torch.xpu.Stream(), "buf_of": {}, "ready": {}, "free": [None, None], "order": None}
        pf = self._pf
        if pf["order"] is None or len(pf["order"]) != len(self.layer_index):
            pf["order"] = list(self.layer_index)
            # model layer order, NOT insertion order (layers are packed as their shards finish loading, out of order)
            import re as _re

            def _lid(k):
                m = _re.search(r"layers\.(\d+)\.", str(k))
                return int(m.group(1)) if m else 1 << 30
            pf["layers"] = sorted((k for k in pf["order"] if not str(k).startswith("mtp.")), key=_lid)
        cs = torch.xpu.current_stream()
        li = pf["order"].index(key)
        if pf["layers"] and key == pf["layers"][0]:
            # first MoE layer of a prefill forward: the device cache may have changed since the last staging -- decode
            # runs as XPU graph replays (no Python here, so no mirror refresh) and SGLang's overlap scheduler can
            # still have decode work queued. Sync, and if the cache's LRU tick moved, rebuild the host mirror of the
            # slot map and drop pre-staged tables (their slot pointers may be stale). Within one prefill forward every
            # MoE call is staged, so nothing else touches the cache until the forward ends.
            torch.xpu.synchronize()
            t = int(self.tick.item())
            if pf.get("tick") != t or self._cache_used:
                self.sync_mirror()
                self._cache_used = False
                pf["buf_of"].clear()
                pf["ready"].clear()
                pf["tick"] = t
        if key not in pf["buf_of"]:
            used = {b for b, _ in pf["buf_of"].values()}
            self._stage_into(key, 0 if 0 not in used else 1)
        buf, table = pf["buf_of"].pop(key)
        rdy = pf["ready"].pop(key)
        if os.environ.get("EXL3_STAGE_VERIFY", "0") == "1":
            import sys as _sys
            print(f"EXL3_STAGE_LOG compute {key} from buf {buf} rows {x.shape[0]}", file=_sys.stderr, flush=True)
        if os.environ.get("EXL3_STAGE_HOSTWAIT", "0") == "1":
            rdy.synchronize()                 # host waits for the copy-engine transfer, then enqueues the compute
        cs.wait_event(rdy)
        if os.environ.get("EXL3_STAGE_CHECK", "0") == "1":
            import sys as _sys
            li_ = self.layer_index[key]
            devt = self.ptrs_all[li_ * self.E:(li_ + 1) * self.E]
            stg = s64(self._stage[buf].data_ptr())
            in_stage = (table >= stg) & (table < stg + self.E * self.blob)
            mism = (~in_stage) & (table != devt)
            sod = self.slot_of_dev[li_ * self.E:(li_ + 1) * self.E]
            print(f"EXL3_STAGE_CHECK {key}: staged-from-slot {int((~in_stage).sum())}, device-table-in-slot "
                  f"{int(((devt >= self.slot_base) & (devt < self.slot_base + self.n_slots * self.blob)).sum())}, "
                  f"slot_of_dev>=0 {int((sod >= 0).sum())}, mismatching slot pointers {int(mism.sum())}",
                  file=_sys.stderr, flush=True)
        ids, w = self._rt(topk_ids, topk_w)
        y = self.X.moe_forward(x, ids, w, table, self.I, self.K, self.E)
        if os.environ.get("EXL3_STAGE_VERIFY", "0") == "1":
            import sys as _sys
            y2 = self.X.moe_forward(x, ids, w, self._host_ptr[key].to(self.dev), self.I, self.K, self.E)
            torch.xpu.synchronize()
            li_ = self.layer_index[key]
            stg = s64(self._stage[buf].data_ptr())
            tab = table.cpu()
            bad_e = []
            for e in range(self.E):
                p_ = int(tab[e])
                if stg <= p_ < stg + self.E * self.blob:
                    if not torch.equal(self._stage[buf][e].cpu(), self.host[key][e]):
                        diff = (self._stage[buf][e].cpu() != self.host[key][e]).nonzero().flatten()
                        bad_e.append((e, int(diff.numel()), int(diff[0]), int(diff[-1])))
                else:
                    s_ = (p_ - self.slot_base) // self.blob
                    if not torch.equal(self.slots[s_].cpu(), self.host[key][e]):
                        bad_e.append(("slot", e, s_))
            print(f"EXL3_STAGE_VERIFY {key} rows {x.shape[0]} equal {torch.equal(y, y2)} "
                  f"maxdiff {(y.float() - y2.float()).abs().max().item():.3g} corrupt {len(bad_e)} {bad_e[:4]}",
                  file=_sys.stderr, flush=True)
        ev = torch.xpu.Event()
        ev.record(cs)
        pf["free"][buf] = ev                          # buffer reusable once this layer's kernels are done
        # stage the next regular layer (wrapping to the first: chunked prefill runs forwards back to back); a pre-staged
        # table is dropped by prefill_reset() as soon as any non-staged MoE call (decode) could move cache slots
        layers = pf["layers"]
        pos = layers.index(key) if key in layers else -1
        if pos >= 0 and layers:
            nxt = layers[(pos + 1) % len(layers)]
            if nxt not in pf["buf_of"] and nxt != key:
                self._stage_into(nxt, 1 - buf)
        return y

    def _stage_into(self, key, buf: int) -> None:
        pf = self._pf
        st = pf["stream"]
        for k_ in [k_ for k_, (b_, _) in pf["buf_of"].items() if b_ == buf]:
            pf["buf_of"].pop(k_)                  # an unconsumed pre-staged layer in this buffer is about to be overwritten
            pf["ready"].pop(k_, None)
        if pf["free"][buf] is not None:
            st.wait_event(pf["free"][buf])
            pf["free"][buf] = None
        table = self.stage_layer(key, buf, st)       # copies the non-resident experts on the side stream
        if os.environ.get("EXL3_STAGE_VERIFY", "0") == "1":
            import sys as _sys
            print(f"EXL3_STAGE_LOG stage {key} -> buf {buf} missing {int((self.slot_of[key] < 0).sum())}",
                  file=_sys.stderr, flush=True)
        ev = torch.xpu.Event()
        ev.record(st)
        pf["buf_of"][key] = (buf, table)
        pf["ready"][key] = ev

    def prefill_reset(self) -> None:
        """Drop pre-staged layers (their tables hold cache-slot pointers valid only until the next cache update)."""
        if hasattr(self, "_pf") and self._pf["buf_of"]:
            self._pf["buf_of"].clear()
            self._pf["ready"].clear()

    def sync_mirror(self) -> None:
        """Rebuild the CPU mirror (slot_of, slot_owner, free list) from the device cache state. Device->host read of
        the slot map (synchronises the current stream): for static placement ops, never in the decode loop."""
        sk = self.slot_key.cpu()
        sod = self.slot_of_dev.cpu()
        keys = {li: key for key, li in self.layer_index.items()}
        for key, li in self.layer_index.items():
            self.slot_of[key] = sod[li * self.E:(li + 1) * self.E].clone()
        self.slot_owner = [None] * self.n_slots
        free = []
        for sl in range(self.n_slots):
            k = int(sk[sl])
            if k < 0:
                free.append(sl)
            else:
                self.slot_owner[sl] = (keys[k // self.E], k % self.E)
        self.free_slots = free[::-1]

    def _resync(self) -> None:
        if self._cache_used:
            self.sync_mirror()
            self._cache_used = False

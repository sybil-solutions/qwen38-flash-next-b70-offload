# N111: VRAM victim ring (kernel a9 + nvtier.NvTierAsyncRing) - correctness and microbench, standalone (no server).
# Mounts: /pkg/exl3xpu = N104 src/exl3xpu (+ N111 nvtier.py on top), /s = NVMe store (ro), /t = K02 traces (ro), /o = out.
# Phases (env PHASES):
#   exact : kernel level. Same cache state, one moe_forward_cached per call in modes plain / wb (direct write-back) /
#           ring (a9 lib only). Outputs vs a golden all-VRAM moe_forward (bit-exact), wb: victim page == data + flag,
#           ring: ring slot == data, flag NOT set, page untouched. Outputs saved for the a8-vs-a9 cross-lib compare.
#   sim   : end-to-end with TierStore + NvTier classes (wb = NvTierAsync, ring = NvTierAsyncRing, novict), small VRAM
#           and RAM tiers -> many evictions. Every call's output vs golden (post-mask ids), pages checked at the moment
#           the host flags a ring write-back resident (no sync), periodic page/ring audits.
#   bench : per-step GPU cost of 48 decode calls (M=1) with V victims/step: none / wb (punched memfd) / ring (+ D2H drain).
import os, sys, json, time, ctypes, mmap, gc
import numpy as np, torch
sys.path.insert(0, "/pkg")
from exl3xpu.moe_offload import ops, s64

X = ops(); dev = torch.device("xpu")
RING_LIB = hasattr(X, "moe_set_victim_ring")
LIBTAG = os.environ.get("LIBTAG", "a9" if RING_LIB else "a8")
OUT = os.environ.get("OUT", f"/o/n111_{LIBTAG}.json")
PHASES = os.environ.get("PHASES", "exact,sim,bench").split(",")
H, I, K, E, TOPK = 2560, 640, 3, 512, 10
REC = 1863680; MiB2 = 2 << 20
BLOB = int(X.blob_bytes(H, I, K)); assert BLOB == 1862400
LAYERS = [6, 18, 30, 42]; NLY = len(LAYERS); LT = NLY + 1; S = NLY * E
KR = int(os.environ.get("KR", "128"))
TBUDGET = float(os.environ.get("TBUDGET", "780"))
T0 = time.perf_counter()
rng = np.random.default_rng(0); torch.manual_seed(0)
res = dict(meta=dict(lib=os.environ.get("EXL3_MOE_LIB"), ring_lib=RING_LIB, device=torch.xpu.get_device_name(0)),
           exact=[], sim=[], bench=[])


def dump():
    with open(OUT + ".tmp", "w") as f: json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)


def log(key, d):
    d["t"] = round(time.perf_counter() - T0, 1)
    print(json.dumps(d), flush=True); res[key].append(d); dump()


def sync(): torch.xpu.synchronize()
def evt(): return torch.xpu.Event(enable_timing=True)
def over_budget(): return time.perf_counter() - T0 > TBUDGET


libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
def u8view(addr, n): return np.ctypeslib.as_array((ctypes.c_uint8 * n).from_address(addr))

# ------------------------------------------------------------------ golden data: the 4 bench layers from the NVMe store
t = time.perf_counter()
nfd = os.open("/s/qwen_experts.bin", os.O_RDONLY | os.O_DIRECT)
gmm = mmap.mmap(-1, S * REC)
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(16)
def rd_gold(k):
    b = (ctypes.c_char * REC).from_buffer(gmm, k * REC)
    n = os.preadv(nfd, [b], (LAYERS[k // E] * E + k % E) * REC); assert n == REC, n
list(pool.map(rd_gold, range(S)))
G = np.frombuffer(gmm, dtype=np.uint8).reshape(S, REC)
gold = torch.from_numpy(G).to(dev)                      # [S, REC] golden VRAM copy (REC stride, 64 B aligned)
GOLD_BASE = s64(gold.data_ptr())
gold_tab = [torch.tensor([GOLD_BASE + (li * E + e) * REC for e in range(E)], dtype=torch.int64, device=dev) for li in range(NLY)]
res["meta"]["gold_load_s"] = round(time.perf_counter() - t, 1)

# ------------------------------------------------------------------ routing (K02 traces)
TR = {}
if os.path.isdir("/t"):
    for f in sorted(os.listdir("/t")):
        if f.endswith(".npz"):
            try:
                z = np.load(os.path.join("/t", f)); TR[f[:-4]] = dict(di=z["decode_ids"], dw=z["decode_w"])
            except Exception as e: print("trace fail", f, e)
DEC = [k for k in sorted(TR) if TR[k]["di"].shape[0] >= 300]
res["meta"]["decode_traces"] = {k: int(TR[k]["di"].shape[0]) for k in DEC}


def route(B, L, s):
    if DEC:
        ids, w = [], []
        for b in range(B):
            tr = TR[DEC[b % len(DEC)]]; n = tr["di"].shape[0]; i = (100 + s + 37 * (b // len(DEC))) % n
            ids.append(tr["di"][i, L]); w.append(tr["dw"][i, L])
        return np.stack(ids).astype(np.int32), np.stack(w).astype(np.float32)
    ids = np.stack([rng.choice(E, TOPK, replace=False) for _ in range(B)]); return ids.astype(np.int32), rng.random((B, TOPK)).astype(np.float32)


XPOOL = [torch.randn(4, H, dtype=torch.bfloat16, device=dev) for _ in range(64)]
dump(); print(json.dumps(res["meta"]), flush=True)

# ------------------------------------------------------------------ kernel-level state (phases exact / bench)
SVM_SIZE = LT * E * MiB2
mfd = os.memfd_create("n111", 0); os.ftruncate(mfd, SVM_SIZE)
resv = libc.mmap(None, SVM_SIZE + MiB2, 0, 0x22, -1, 0)
SVM_BASE = (resv + MiB2 - 1) // MiB2 * MiB2
assert libc.mmap(ctypes.c_void_p(SVM_BASE), SVM_SIZE, 3, 0x01 | 0x10, mfd, 0) == SVM_BASE
libc.madvise(SVM_BASE, SVM_SIZE, 14)
def svm_addr(key): return SVM_BASE + key * MiB2
def punch(keys):
    for k in keys:
        r = libc.fallocate(mfd, 3, k * MiB2, MiB2); assert r == 0, ctypes.get_errno()
for k in range(S): ctypes.memmove(svm_addr(k), G[k].ctypes.data, BLOB)
DUMMY0 = S
slots = None; SLOT_BASE = 0
ptrs_all = torch.zeros(LT * E, dtype=torch.int64, device=dev)
slot_of = torch.full((LT * E,), -1, dtype=torch.int32, device=dev)
slot_key = torch.full((S,), -1, dtype=torch.int32, device=dev)
slot_last = torch.full((S,), -1, dtype=torch.int32, device=dev)
tick = torch.ones(1, dtype=torch.int32, device=dev)
host_base = torch.tensor([s64(SVM_BASE + li * E * MiB2) for li in range(LT)], dtype=torch.int64, device=dev)
fill_all = torch.zeros(LT * E, dtype=torch.int64, device=dev)
fill_list = torch.zeros(1 + E, dtype=torch.int32, device=dev)
pend = torch.zeros(0, dtype=torch.int32, device=dev); done_seq = torch.zeros(0, dtype=torch.int32, device=dev)
ram_res = torch.zeros(LT * E, dtype=torch.uint8, device=dev)
mring = torch.zeros(1 + 65536, dtype=torch.int32, device=dev)
safe = torch.zeros(LT, dtype=torch.int32, device=dev)
vict = torch.zeros(1 + 2 * E, dtype=torch.int32, device=dev)
vr_meta = torch.zeros(4 + 2 * KR, dtype=torch.int32, device=dev)
vr_sel = torch.full((E,), -1, dtype=torch.int32, device=dev)
vr_buf = torch.empty(KR * BLOB, dtype=torch.uint8, device=dev); VR_BASE = s64(vr_buf.data_ptr())
EMPTY_I = torch.zeros(0, dtype=torch.int32, device=dev); EMPTY_U = torch.zeros(0, dtype=torch.uint8, device=dev)


def ensure_slots():
    global slots, SLOT_BASE
    if slots is None:
        slots = torch.empty((S, BLOB), dtype=torch.uint8, device=dev); SLOT_BASE = s64(slots.data_ptr())
def slot_ptr(k): return SLOT_BASE + k * BLOB


def reset_ring():
    m = torch.full((4 + 2 * KR,), -1, dtype=torch.int32); m[:4] = torch.tensor([0, 0, 0, KR], dtype=torch.int32); m[4 + KR:] = 0
    vr_meta.copy_(m.to(dev))


def configure(mode):
    mode = "ring" if mode == "ringT" else mode
    """plain: no tier; none: tier, no victims; wb: tier + direct write-back; ring: tier + victims via the VRAM ring"""
    X.moe_set_host_stride(MiB2)
    if mode == "plain": X.moe_set_tier(EMPTY_U, EMPTY_I, EMPTY_I)
    else: X.moe_set_tier(ram_res, mring, safe)
    X.moe_set_victims(vict if mode in ("wb", "ring") else EMPTY_I)
    if RING_LIB:
        if mode == "ring": reset_ring(); X.moe_set_victim_ring(vr_meta, vr_sel, vr_buf)
        else: X.moe_set_victim_ring(EMPTY_I, EMPTY_I, EMPTY_U)


def build_state(miss_keys, dummies):
    """all real keys in slot k (age 1); miss keys host-only (RAM-resident), their slots owned by dummy keys (age 0,
    ram_res 0) -> the dummies are the LRU victims, written back (wb/ring) to their own RAM-tier address."""
    n = LT * E
    pa = np.array([slot_ptr(k) if k < S else s64(svm_addr(k)) for k in range(n)], dtype=np.int64)
    so = np.where(np.arange(n) < S, np.arange(n), -1).astype(np.int32)
    sk = np.arange(S, dtype=np.int32); sl = np.ones(S, dtype=np.int32)
    rr = np.zeros(n, dtype=np.uint8); rr[:S] = 1
    for k, d in zip(miss_keys, dummies):
        pa[k] = s64(svm_addr(k)); so[k] = -1
        sk[k] = d; so[d] = k; pa[d] = slot_ptr(k); sl[k] = 0
    tt = lambda a: torch.from_numpy(a).to(dev)
    return tt(pa), tt(so), tt(sk), tt(sl), tt(rr)


def apply_state(st):
    pa, so, sk, sl, rr = st
    ptrs_all.copy_(pa); slot_of.copy_(so); slot_key.copy_(sk); slot_last.copy_(sl); ram_res.copy_(rr); tick.fill_(1)


def fcached(li, xs, di, dw):
    return X.moe_forward_cached(xs, di, dw, ptrs_all, li, slot_of, slot_key, slot_last, tick, host_base, SLOT_BASE,
                                fill_all, fill_list, E, I, K, E, pend, done_seq)


def reload_slots():
    slots.copy_(gold[:, :BLOB])


def pick_miss(ids, n):
    u = np.unique(ids); return np.sort(rng.choice(u, min(n, len(u)), replace=False))


# ------------------------------------------------------------------ phase exact
if "exact" in PHASES:
    ensure_slots(); reload_slots(); sync()
    modes = ["plain", "wb"] + (["ring"] if RING_LIB else [])
    outs = {m: [] for m in modes}
    agg = {m: dict(calls=0, gold_exact=0, victims=0, wb_page_ok=0, wb_flag_ok=0, ring_slot_ok=0, ring_flag_untouched=0,
                   ring_page_untouched=0, ring_meta_ok=0, fills_ok=0) for m in modes}
    for M in (1, 2, 4):
        for li in range(NLY):
            for s in range(6):
                ids, w = route(M, LAYERS[li], 50 + s)
                miss = pick_miss(ids, 3 + s)                         # 3..8 misses (victims) per call
                mk = [li * E + int(e) for e in miss]
                dums = [DUMMY0 + (li * 64 + s * 8 + j) % E for j in range(len(mk))]
                xs = XPOOL[(M * 13 + li * 7 + s) % 64][:M].contiguous()
                for mode in modes:
                    configure(mode)
                    punch(dums)
                    reload_slots()
                    st = build_state(mk, dums); apply_state(st)
                    di, dw = torch.from_numpy(ids).to(dev), torch.from_numpy(w).to(dev)
                    y = fcached(li, xs, di, dw)
                    yr = X.moe_forward(xs, di, dw, gold_tab[li], I, K, E)
                    sync()
                    a = agg[mode]; a["calls"] += 1; a["victims"] += len(mk)
                    a["gold_exact"] += int(torch.equal(y, yr))
                    outs[mode].append(y.view(torch.int16).cpu().numpy().ravel())
                    so = slot_of.cpu().numpy(); a["fills_ok"] += int(all(so[k] == k for k in mk))
                    rrh = ram_res.cpu().numpy()
                    if mode == "wb":
                        a["wb_page_ok"] += sum(int(np.array_equal(u8view(svm_addr(d), BLOB), G[k, :BLOB])) for k, d in zip(mk, dums))
                        a["wb_flag_ok"] += sum(int(rrh[d] == 1) for d in dums)
                    if mode == "ring":
                        vm = vr_meta.cpu().numpy(); vk = vm[4:4 + KR]; sel = vr_sel.cpu().numpy(); vl = vict.cpu().numpy()
                        rb = vr_buf.view(KR, BLOB)
                        for i in range(int(vl[0])):                  # victim entry i: (old key, slot) -> ring slot sel[i]
                            old, sl_, r = int(vl[1 + 2 * i]), int(vl[2 + 2 * i]), int(sel[i])
                            if old < 0: continue
                            a["ring_meta_ok"] += int(r >= 0 and vk[r] == old and old in dums)
                            a["ring_slot_ok"] += int(r >= 0 and torch.equal(rb[r].cpu(), torch.from_numpy(G[sl_, :BLOB])))
                        for d in dums:
                            a["ring_flag_untouched"] += int(rrh[d] == 0)
                            a["ring_page_untouched"] += int(not u8view(svm_addr(d), BLOB).any())
    for m in modes:
        o = np.concatenate(outs[m]); np.save(f"/o/exact_{LIBTAG}_{m}.npy", o)
        d = dict(mode=m, **agg[m])
        if m != "plain": d["equal_to_plain"] = bool(np.array_equal(o, np.concatenate(outs["plain"])))
        log("exact", d)
    configure("plain")

# ------------------------------------------------------------------ phase sim (needs a9 for ring)
if "sim" in PHASES:
    import exl3xpu.nvtier as NT
    meta = json.load(open("/s/qwen_experts.json"))
    SIM_MODES = os.environ.get("SIM_MODES", "wb,ring,novict" if RING_LIB else "wb,novict").split(",")
    SIM_SLOTS = int(os.environ.get("SIM_SLOTS", "640")); SIM_RAM = int(os.environ.get("SIM_RAM", "640"))
    SIM_STEPS = int(os.environ.get("SIM_STEPS", "1200")); SIM_M = [int(v) for v in os.environ.get("SIM_M", "1,4").split(",")]
    SIM_KS = [int(v) for v in os.environ.get("SIM_KS", "128,16").split(",")]
    CHECK_EVERY = 100
    if slots is not None:
        del slots; slots = None; gc.collect(); torch.xpu.empty_cache()
    runs = []
    for M in SIM_M:
        for mode in SIM_MODES:
            for kr in (SIM_KS if mode == "ring" else [0]):
                runs.append((M, mode, kr))
    for M, mode, kr in runs:
        if over_budget(): break
        os.environ["EXL3_NVTIER_NO_VICTIM"] = "1" if mode == "novict" else "0"
        os.environ["EXL3_NVTIER_VRING_K"] = str(kr or 128)
        store = NT.TierStore(H, I, K, E, SIM_SLOTS, n_layers=NLY)
        for li in range(NLY): store.add_layer(f"l{li}")
        cls = NT.NvTierAsyncRing if mode == "ring" else NT.NvTierAsync
        tier = cls(store, "/s/qwen_experts.bin", meta, SIM_RAM * MiB2, ckpt_layer=LAYERS)
        sync()
        bad_out = torch.zeros(1, dtype=torch.int64, device=dev)
        chk = dict(land_checked=0, land_bad=0, audit_pages=0, audit_bad=0, ring_checked=0, ring_bad=0)
        steps_n = SIM_STEPS if M == 1 else SIM_STEPS // 2
        t = time.perf_counter()
        for s in range(steps_n):
            for li in range(NLY):
                ids, w = route(M, LAYERS[li], s)
                di, dw = torch.from_numpy(ids).to(dev), torch.from_numpy(w).to(dev)
                xs = XPOOL[(s * 4 + li) % 64][:M].contiguous()
                y = X.moe_forward_cached(xs, di, dw, store.ptrs_all, li, store.slot_of_dev, store.slot_key, store.slot_last,
                                         store.tick, store.host_base, store.slot_base, store.fill_all, store.fill_list,
                                         E, I, K, E, store.pend, store.done_seq)
                yr = X.moe_forward(xs, di, dw, gold_tab[li], I, K, E)      # di/dw now post-mask (ensure wrote them)
                bad_out += (y != yr).any(dim=1).sum()
            if mode == "ring":
                before = {id(fe): items for fe, items in tier._vr_fly}
                tier.step()
                after = {id(fe) for fe, _ in tier._vr_fly}
                landed = [kk for fid, items in before.items() if fid not in after for _, kk in items]
                for kk in landed:                            # flagged resident just now: data must be there already
                    chk["land_checked"] += 1
                    if not np.array_equal(u8view(store.addr(kk), BLOB), G[kk, :BLOB]): chk["land_bad"] += 1
            else:
                tier.step()
            if (s + 1) % CHECK_EVERY == 0:
                sync()
                rr = tier.res_dev.cpu().numpy()
                resident = np.nonzero(rr[:NLY * E] == 1)[0]
                samp = resident if len(resident) <= 256 else rng.choice(resident, 256, replace=False)
                for kk in samp.tolist():
                    chk["audit_pages"] += 1
                    if not np.array_equal(u8view(store.addr(kk), BLOB), G[kk, :BLOB]): chk["audit_bad"] += 1
                if mode == "ring":
                    vm = tier.vr_meta.cpu().numpy(); vk = vm[4:4 + tier.vr_K]; rb = tier.vr_buf.view(tier.vr_K, BLOB)
                    for r in np.nonzero(vk >= 0)[0].tolist():   # every claimed ring slot holds its key's data
                        chk["ring_checked"] += 1
                        if not torch.equal(rb[r].cpu(), torch.from_numpy(G[int(vk[r]), :BLOB])): chk["ring_bad"] += 1
        sync(); wall = time.perf_counter() - t
        st = tier.stats; n = max(1, st["steps"])
        d = dict(M=M, mode=mode, K=kr, steps=st["steps"], slots=SIM_SLOTS, ram_experts=SIM_RAM, out_mismatch_rows=int(bad_out.item()),
                 masked_per_step=round(st["masked"] / n, 2), nvme_per_step=round(st["nvme_reads"] / n, 2),
                 evicts_per_step=round(st["evicts"] / n, 2), rescued=st.get("evict_rescued", 0), wall_s=round(wall, 1), **chk)
        for kk in ("vr_claims", "vr_drops", "vr_issued", "vr_landed", "vr_deferred"):
            if kk in st: d[kk + "_per_step"] = round(st[kk] / n, 2)
        if mode == "ring": d.update(final_inflight=sum(len(i) for _, i in tier._vr_fly), busy_negative=int((tier.vr_busy < 0).sum()),
                                    ppend_negative=int((tier.ppend < 0).sum()))
        log("sim", d)
        tier.close(); sync()
        libc.fallocate(store.fd, 3, 0, store.size)
        os.close(store.fd)
        del tier, store; gc.collect(); torch.xpu.empty_cache()
    os.environ["EXL3_NVTIER_NO_VICTIM"] = "0"

# ------------------------------------------------------------------ phase bench: per-step cost at V victims / step
if "bench" in PHASES and not over_budget():
    ensure_slots(); reload_slots(); sync()
    VS = [int(v) for v in os.environ.get("VS", "0,10,20,30").split(",")]
    DRAIN_LIB = hasattr(X, "moe_vring_drain_push")
    BMODES = [m for m in os.environ.get("BMODES", "none,wb,ring,ringT,ringU").split(",")
              if (m != "ring" or RING_LIB) and (m != "ringT" or DRAIN_LIB)]
    NST = int(os.environ.get("BSTEPS", "8")); NCALL = 48
    # call c of a step: layer c % 4, trace step 200 + c // 4
    calls = [(c % NLY, *route(1, LAYERS[c % NLY], 200 + c // NLY)) for c in range(NCALL)]
    cd = [(li, torch.from_numpy(ids).to(dev), torch.from_numpy(w).to(dev)) for li, ids, w in calls]
    xs1 = XPOOL[0][:1].contiguous()
    for V in VS:
        # V misses spread over the 48 calls (each miss = one victim), fixed per V so every mode sees the same work
        per_call = np.bincount(rng.choice(NCALL, V, replace=True), minlength=NCALL) if V else np.zeros(NCALL, dtype=np.int64)
        plan = []
        dctr = 0
        for c, (li, ids, w) in enumerate(calls):
            m = pick_miss(ids, int(per_call[c]))
            mk = [li * E + int(e) for e in m]
            dums = [DUMMY0 + (dctr + j) % E for j in range(len(mk))]; dctr += len(mk)
            plan.append((mk, dums, build_state(mk, dums)))
        for mode in BMODES:
            if over_budget(): break
            configure(mode)
            # warm: map the miss pages on the GPU (they are read zero-copy)
            for (li, di, dw), (mk, dums, st) in zip(cd, plan): apply_state(st); fcached(li, xs1, di.clone(), dw.clone())
            sync(); reload_slots(); sync()
            gpu, wall, d2h_ms, lag, drained, issue = [], [], [], [], 0, []
            RM = mode in ("ring", "ringT", "ringU")
            if mode == "ringU":
                stage = X.host_alloc(KR * BLOB); SB = s64(stage.data_ptr())
                from concurrent.futures import ThreadPoolExecutor as _TPE
                cpool = _TPE(2); stg = []
            cs = torch.xpu.Stream(); fly = []; seen = np.zeros(KR, dtype=np.int64); snap_prev = None
            for step in range(NST + 2):
                cs.synchronize()                                       # (ring) last step's D2H done before re-punching its targets
                if mode == "ringU":
                    cs.synchronize()
                    while stg or fly:
                        for b, rs, ks, st0 in stg:
                            for r, kk in zip(rs, ks): ctypes.memmove(svm_addr(kk), SB + r * BLOB, BLOB)
                            free_ = rs
                            vr_meta.index_fill_(0, torch.tensor(rs, dtype=torch.int64).add_(4).to(dev), -1); drained += len(rs)
                        stg = []
                        for a, b, rs, st0 in fly:
                            a.result(); vr_meta.index_fill_(0, torch.tensor(rs, dtype=torch.int64).add_(4).to(dev), -1); drained += len(rs)
                        fly = []
                if mode == "ringT":
                    while fly and X.moe_vring_drain_done() < max(a for a, _, _, _ in fly): time.sleep(0.0002)
                punch([d for _, dums, _ in plan for d in dums])       # write-back targets: freshly punched (realistic)
                tw = time.perf_counter()
                if RM and snap_prev is not None:
                    hv, sev = snap_prev; sev.synchronize(); vm = hv.numpy()
                    vk, vs = vm[4:4 + KR], vm[4 + KR:]
                    new = np.nonzero((vk >= 0) & (vs != seen))[0]
                    ti = time.perf_counter()
                    if len(new) and mode == "ringU":
                        with torch.xpu.stream(cs):
                            cs.wait_event(sev)
                            for r in new.tolist():
                                X.memcpy_async(SB + r * BLOB, VR_BASE + r * BLOB, BLOB); seen[r] = vs[r]
                            b = evt(); b.record(cs)
                        stg.append((b, new.tolist(), [int(vk[r]) for r in new.tolist()], step))
                    if mode == "ringU":
                        k2 = []
                        for b, rs, ks, st0 in stg:
                            if b.query():
                                def _cp(rs=rs, ks=ks):
                                    for r, kk in zip(rs, ks): ctypes.memmove(svm_addr(kk), SB + r * BLOB, BLOB)
                                fly.append((cpool.submit(_cp), "F", rs, st0))
                            else: k2.append((b, rs, ks, st0))
                        stg = k2
                    if len(new) and mode == "ringT":
                        for r in new.tolist():
                            sq = X.moe_vring_drain_push(VR_BASE + r * BLOB, s64(svm_addr(int(vk[r]))), BLOB); seen[r] = vs[r]
                        fly.append((sq, None, new.tolist(), step))
                    elif len(new):
                        with torch.xpu.stream(cs):
                            cs.wait_event(sev)
                            a = evt(); a.record(cs)
                            for r in new.tolist():
                                X.memcpy_async(s64(svm_addr(int(vk[r]))), VR_BASE + r * BLOB, BLOB); seen[r] = vs[r]
                            b = evt(); b.record(cs)
                        fly.append((a, b, new.tolist(), step))
                    issue.append((time.perf_counter() - ti) * 1e3)
                    keep, free = [], []
                    dn = X.moe_vring_drain_done() if mode == "ringT" else 0
                    for a, b, rs, st0 in fly:
                        if (a.done() if b == "F" else (a <= dn) if b is None else b.query()):
                            free += rs; lag.append(step - st0); drained += len(rs)
                            if b is not None and b != "F": d2h_ms.append(a.elapsed_time(b) / max(1, len(rs)))
                        else: keep.append((a, b, rs, st0))
                    fly = keep
                    if free:
                        vr_meta.index_fill_(0, torch.tensor(free, dtype=torch.int64).add_(4).to(dev, non_blocking=True), -1)
                evs = []
                for (li, di, dw), (mk, dums, st) in zip(cd, plan):
                    apply_state(st); di2, dw2 = di.clone(), dw.clone()
                    a, b = evt(), evt(); a.record(); fcached(li, xs1, di2, dw2); b.record(); evs.append((a, b))
                if RM:
                    hv = torch.empty(4 + 2 * KR, dtype=torch.int32).pin_memory(); hv.copy_(vr_meta, non_blocking=True)
                    sev = torch.xpu.Event(); sev.record(); snap_prev = (hv, sev)
                torch.xpu.current_stream().synchronize()
                tw = time.perf_counter() - tw
                if step >= 2:
                    gpu.append(sum(a.elapsed_time(b) for a, b in evs)); wall.append(tw * 1e3)
            sync()
            vm = vr_meta.cpu().numpy() if RM else None
            d = dict(V=V, mode=mode, steps=len(gpu), gpu_ms_per_step=round(float(np.mean(gpu)), 3),
                     gpu_ms_median=round(float(np.median(gpu)), 3), wall_ms_per_step=round(float(np.mean(wall)), 3),
                     victims_per_step=int(sum(len(mk) for mk, _, _ in plan)))
            if RM:
                d.update(issue_ms_per_step=round(float(np.mean(issue)), 3) if issue else None, d2h_ms_per_blob=round(float(np.mean(d2h_ms)), 3) if d2h_ms else None, drain_lag_steps=round(float(np.mean(lag)), 2) if lag else None,
                         drained=drained, drops=int(vm[2]), claims=int(vm[1]))
            log("bench", d)
    # derived: per-step overhead vs none
    for V in VS:
        b = {r["mode"]: r["gpu_ms_per_step"] for r in res["bench"] if r["V"] == V}
        if "none" in b:
            log("bench", dict(V=V, summary=True, **{f"{m}_minus_none_ms": round(v - b["none"], 3) for m, v in b.items() if m != "none"}))
    configure("plain")

res["meta"]["total_s"] = round(time.perf_counter() - T0, 1)
dump()
print("DONE", flush=True)

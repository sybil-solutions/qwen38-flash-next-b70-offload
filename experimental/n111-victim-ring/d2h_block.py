# N111: is memcpy_async (VRAM -> host) blocking the calling host thread? dst = USM host / memfd warm / memfd punched
import os, sys, time, ctypes, json
import torch
sys.path.insert(0, "/pkg")
from exl3xpu.moe_offload import ops, s64
X = ops(); dev = torch.device("xpu"); BLOB = 1862400; MiB2 = 2 << 20; N = 30
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long, ctypes.c_long]
SZ = N * MiB2; fd = os.memfd_create("d2h", 0); os.ftruncate(fd, SZ)
resv = libc.mmap(None, SZ + MiB2, 0, 0x22, -1, 0); base = (resv + MiB2 - 1) // MiB2 * MiB2
assert libc.mmap(ctypes.c_void_p(base), SZ, 3, 0x11, fd, 0) == base; libc.madvise(base, SZ, 14)
usm = X.host_alloc(N * BLOB); ub = s64(usm.data_ptr())
src = torch.randint(0, 255, (N * BLOB,), dtype=torch.uint8, device=dev); sb = s64(src.data_ptr())
cs = torch.xpu.Stream(); out = []
for case in ("usm", "memfd_warm", "memfd_punched", "usm", "memfd_punched"):
    for rep in range(3):
        if case == "memfd_punched": libc.fallocate(fd, 3, 0, SZ)
        if case == "memfd_warm": ctypes.memset(base, 1, SZ)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        with torch.xpu.stream(cs):
            for i in range(N):
                X.memcpy_async(ub + i * BLOB if case == "usm" else base + i * MiB2, sb + i * BLOB, BLOB)
            ev = torch.xpu.Event(); ev.record(cs)
        t1 = time.perf_counter(); ev.synchronize(); t2 = time.perf_counter()
        ok = bytes((ctypes.c_char * 64).from_address(ub if case == "usm" else base)) == bytes(src[:64].cpu().numpy())
        d = dict(case=case, rep=rep, issue_ms=round((t1 - t0) * 1e3, 2), total_ms=round((t2 - t0) * 1e3, 2), ok=ok)
        print(json.dumps(d), flush=True); out.append(d)
json.dump(out, open("/o/d2h_block.json", "w"), indent=1)

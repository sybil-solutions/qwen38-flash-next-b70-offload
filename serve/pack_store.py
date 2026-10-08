#!/usr/bin/env python3
"""Packed NVMe expert store for Qwen3.8-Flash-Next EXL3 3.05bpw_h5_ng5 (the N104 layout).

One 4K-aligned record per (layer, expert): record r = layer*512 + e at offset r*REC, holding the exl3xpu blob
(pack_expert of gate/up/down trellis + suh/svh, K=3); 48 x 512 x 1,863,680 B = 45.8 GB, plus qwen_experts.json (meta)
and qwen_experts.bin.done.

  pack_store.py pack   [--model DIR] [--out DIR]   build (resumable only as a whole: rebuilt unless .done), verify 64
  pack_store.py verify [--model DIR] [--out DIR]   size + .done + O_DIRECT re-read of 64 random records vs a fresh pack
  pack_store.py probe  [--out DIR]                 the store exists and O_DIRECT reads work on its filesystem
"""
import argparse, json, mmap, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["pack", "verify", "probe"])
ap.add_argument("--model", default=os.environ.get("EXL3_MODEL_PATH", "/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5"))
ap.add_argument("--out", default=os.path.dirname(os.environ.get("EXL3_NVTIER_STORE", "/nvx/qwen_experts.bin")))
a = ap.parse_args()
L, E, K = 48, 512, 3
BIN, META = os.path.join(a.out, "qwen_experts.bin"), os.path.join(a.out, "qwen_experts.json")


def odirect_probe(path, nbytes=1 << 20):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    except OSError as e:
        sys.exit(f"O_DIRECT open failed on {path}: {e} (the store must live on a local filesystem that supports O_DIRECT, e.g. xfs or ext4)")
    buf = mmap.mmap(-1, nbytes)
    try:
        n = os.preadv(fd, [buf], 0)
    except OSError as e:
        sys.exit(f"O_DIRECT read failed on {path}: {e}")
    finally:
        os.close(fd)
    return n


def packer():
    import torch  # noqa: F401
    from safetensors import safe_open
    from exl3xpu.moe_offload import pack_expert
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]; fh = {}
    def get(n):
        f = idx[n]
        if f not in fh: fh[f] = safe_open(os.path.join(a.model, f), "pt")
        return fh[f].get_tensor(n)
    def pack(l, e):
        pre = f"model.language_model.layers.{l}.mlp.experts.{e}."
        d = {n: {k: get(pre + n + "." + k) for k in ("trellis", "suh", "svh")} for n in ("gate_proj", "up_proj", "down_proj")}
        return pack_expert(d["gate_proj"], d["up_proj"], d["down_proj"], K).numpy().tobytes()
    return pack


def verify(pack, n=64):
    meta = json.load(open(META)); blob, rec = meta["blob"], meta["rec"]
    assert os.path.getsize(BIN) == L * E * rec, "store size mismatch"
    fd = os.open(BIN, os.O_RDONLY | os.O_DIRECT); buf = mmap.mmap(-1, rec); bad = 0
    random.seed(0)
    for _ in range(n):
        l, e = random.randrange(L), random.randrange(E)
        assert os.preadv(fd, [buf], (l * E + e) * rec) == rec
        bad += bytes(buf[:blob]) != pack(l, e)
    os.close(fd)
    return bad


if a.cmd == "probe":
    if not (os.path.exists(BIN) and os.path.exists(BIN + ".done") and os.path.exists(META)):
        sys.exit(f"expert store missing in {a.out} (qwen_experts.bin + .done + qwen_experts.json): run the image with `pack-store` first")
    odirect_probe(BIN); print(json.dumps({"store": BIN, "o_direct": True})); sys.exit(0)

os.makedirs(a.out, exist_ok=True)
pack = packer()
if a.cmd == "pack":
    t0 = time.time()
    blob0 = pack(0, 0); BLOB = len(blob0); REC = (BLOB + 4095) // 4096 * 4096
    if not (os.path.exists(BIN) and os.path.getsize(BIN) == L * E * REC and os.path.exists(BIN + ".done")):
        fd = os.open(BIN, os.O_RDWR | os.O_CREAT, 0o644); os.ftruncate(fd, L * E * REC)
        pad = bytes(REC - BLOB)
        for l in range(L):
            for e in range(E):
                os.pwrite(fd, pack(l, e) + pad, (l * E + e) * REC)
            if l % 8 == 7: print(json.dumps({"layer": l, "elapsed_s": round(time.time() - t0)}), flush=True)
        os.fsync(fd); os.close(fd)
        json.dump({"L": L, "E": E, "K": K, "blob": BLOB, "rec": REC, "H": 2560, "I": 640}, open(META, "w"))
        open(BIN + ".done", "w").close()
    odirect_probe(BIN)
bad = verify(pack)
print(json.dumps({"store": BIN, "verify_bad": bad, "verify_n": 64}), flush=True)
sys.exit(1 if bad else 0)

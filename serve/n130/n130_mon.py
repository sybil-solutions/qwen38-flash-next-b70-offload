# usage: python3 n130_mon.py <out.jsonl> <container> <cardN>   1 s samples: md127 + nvme reads, B70 gt idle residency, container cgroup memory
import json, os, sys, time, subprocess, glob
out, name, card = sys.argv[1], sys.argv[2], sys.argv[3]
DEVS = ["md127", "nvme0n1", "nvme1n1", "nvme2n1", "nvme3n1"]
def disk():
    r = {}
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] in DEVS: r[f[2]] = (int(f[3]), int(f[5]))      # reads completed, sectors read
    return r
def idle(gt):
    try: return int(open(f"/sys/class/drm/{card}/device/tile0/{gt}/gtidle/idle_residency_ms").read())
    except Exception: return None
cg = None; cid = None
def cgroup():
    global cg, cid
    if cg and os.path.exists(cg): return cg
    try:
        cid = subprocess.run(["docker", "inspect", "-f", "{{.Id}}", name], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception: cid = ""
    p = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope" if cid else None
    cg = p if p and os.path.exists(p) else None
    return cg
def mem():
    p = cgroup()
    if not p: return None
    try:
        st = dict(l.split() for l in open(p + "/memory.stat"))
        ev = dict(l.split() for l in open(p + "/memory.events"))
        r = dict(cur=int(open(p + "/memory.current").read()), anon=int(st["anon"]), shmem=int(st["shmem"]), file=int(st["file"]),
                 file_mapped=int(st["file_mapped"]), anon_thp=int(st.get("anon_thp", 0)), oom_kill=int(ev.get("oom_kill", 0)), oom=int(ev.get("oom", 0)),
                 max_ev=int(ev.get("max", 0)))
        try: r["peak"] = int(open(p + "/memory.peak").read())
        except Exception: pass
        try: r["swap"] = int(open(p + "/memory.swap.current").read()); r["swap_max"] = open(p + "/memory.swap.max").read().strip()
        except Exception: pass
        r["swapcached"] = int(st.get("swapcached", 0)); r["zswap"] = int(st.get("zswap", 0))
        return r
    except Exception: return None
f = open(out, "a")
while True:
    t = time.time()
    rec = dict(t=round(t, 3), disk=disk(), idle0=idle("gt0"), idle1=idle("gt1"), mem=mem())
    f.write(json.dumps(rec) + "\n"); f.flush()
    time.sleep(max(0.05, 1.0 - (time.time() - t)))

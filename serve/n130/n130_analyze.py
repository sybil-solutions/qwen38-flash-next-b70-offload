# usage: python3 n130_analyze.py <run dir>   per-cell NVMe (md127) read GB/s + IOPS, B70 gt0 busy %, cgroup memory peaks
import json, sys, bisect
R = sys.argv[1]; CEIL = 26.0
M = [json.loads(l) for l in open(f"{R}/mon.jsonl")]
T = [m["t"] for m in M]
cells = [json.loads(l) for l in open(f"{R}/matrix.jsonl") if l.strip().startswith("{")]
def win(a, b):
    i, j = bisect.bisect_left(T, a), bisect.bisect_right(T, b) - 1
    if j <= i: return None
    x, y = M[i], M[j]; dt = y["t"] - x["t"]
    dr = y["disk"]["md127"][0] - x["disk"]["md127"][0]; ds = y["disk"]["md127"][1] - x["disk"]["md127"][1]
    busy = None
    if x["idle0"] is not None and y["idle0"] is not None: busy = max(0.0, 1 - (y["idle0"] - x["idle0"]) / (dt * 1e3))
    # peak 1 s read rate inside the window
    pk = 0.0
    for k in range(i + 1, j + 1):
        d = (M[k]["disk"]["md127"][1] - M[k - 1]["disk"]["md127"][1]) * 512 / 1e9 / (M[k]["t"] - M[k - 1]["t"]); pk = max(pk, d)
    mem = [m["mem"] for m in M[i:j + 1] if m["mem"]]
    return dict(s=round(dt, 1), gbs=round(ds * 512 / 1e9 / dt, 2), pct=round(100 * ds * 512 / 1e9 / dt / CEIL, 1), peak_gbs=round(pk, 2),
                iops=round(dr / dt), avg_kb=round(ds * 512 / 1024 / max(dr, 1)), gpu_busy=round(100 * busy, 1) if busy is not None else None,
                mem_peak_gib=round(max(m["cur"] for m in mem) / 2 ** 30, 2) if mem else None,
                anon_peak_gib=round(max(m["anon"] for m in mem) / 2 ** 30, 2) if mem else None,
                shmem_peak_gib=round(max(m["shmem"] for m in mem) / 2 ** 30, 2) if mem else None)
out = []
for c in cells:
    if "t0_abs" not in c: continue
    pf = win(c["t0_abs"], max(c["first_abs"])); dc = win(max(c["first_abs"]), min(c["last_abs"]))
    agg = c.get("decode_agg_overlap") or sum(c["decode_per_stream"])
    r = dict(ctx=c["ctx"], conc=c["conc"], prefill_tok_s=c["prefill_agg_tok_s"], decode=agg if c["conc"] > 1 else c["decode_per_stream"][0],
             decode_per_stream=c["decode_per_stream"], completion=c["completion"], ttft_s=c["ttft_s"], prefill_win=pf, decode_win=dc)
    out.append(r); print(json.dumps(r))
mem = [m["mem"] for m in M if m["mem"]]
summ = dict(mem_peak_gib=round(max(m["cur"] for m in mem) / 2 ** 30, 2), anon_peak_gib=round(max(m["anon"] for m in mem) / 2 ** 30, 2),
            shmem_peak_gib=round(max(m["shmem"] for m in mem) / 2 ** 30, 2), cgroup_peak_gib=round(max(m.get("peak", 0) for m in mem) / 2 ** 30, 2),
            oom_kill=max(m["oom_kill"] for m in mem), swap_max=max(m.get("swap", 0) for m in mem))
print(json.dumps(summ))
json.dump(dict(cells=out, summary=summ), open(f"{R}/analysis.json", "w"), indent=1)

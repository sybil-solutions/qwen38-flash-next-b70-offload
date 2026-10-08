# Prefill/decode matrix at concurrency: distinct prompts per stream (no prefix sharing), natural completion (no caps).
import json, time, threading, urllib.request, sys, random
U = "http://127.0.0.1:30330/v1/chat/completions"
TOPICS = ["SSD wear leveling and garbage collection", "how TCP congestion control evolved", "the history of the printing press",
          "how vaccines train the immune system", "why bridges use expansion joints", "how compilers allocate registers",
          "ocean currents and climate", "how GPS computes position"]
def prompt(ntok, seed):
    rnd = random.Random(seed); words = ["river", "stone", "lamp", "copper", "north", "violet", "engine", "harbor", "maple", "signal"]
    lines, n = [], 0
    while n < ntok:
        lines.append(f"Record {len(lines)}-{seed}: " + " ".join(rnd.choice(words) for _ in range(12)) + f" code {rnd.randint(0, 99999)}.")
        n += 31.7
    return "\n".join(lines) + f"\n\nIgnore the records above. Write a detailed 600-word essay on {TOPICS[seed % len(TOPICS)]}."
def stream(p, out):
    try: _stream(p, out)
    except Exception as e: out["error"] = repr(e)[:200]
def _stream(p, out):
    b = {"model": "flashnext", "temperature": 0, "stream": True, "messages": [{"role": "user", "content": p}],
         "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    t0 = time.time(); ts = []; usage = None
    req = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("usage"): usage = j["usage"]
            if j.get("choices") and j["choices"][0]["delta"].get("content"): ts.append(time.time())
    out.update(t0=t0, ts=ts, prompt=usage["prompt_tokens"], completion=usage["completion_tokens"])
def cell(ctx, conc, seed0):
    outs = [dict() for _ in range(conc)]
    th = [threading.Thread(target=stream, args=(prompt(ctx, seed0 + i), outs[i])) for i in range(conc)]
    for t in th: t.start()
    for t in th: t.join()
    if any("error" in o for o in outs):
        r = dict(ctx=ctx, conc=conc, error=[o.get("error") for o in outs]); print(json.dumps(r), flush=True); return r
    t0 = min(o["t0"] for o in outs)
    ttft = [o["ts"][0] - o["t0"] for o in outs]
    dec = [(o["completion"] - 1) / (o["ts"][-1] - o["ts"][0]) for o in outs]
    lo = max(o["ts"][0] for o in outs); hi = min(o["ts"][-1] for o in outs)      # all streams decoding
    agg = sum(sum(1 for x in o["ts"] if lo <= x <= hi) for o in outs) / (hi - lo) if hi - lo > 5 else None
    tot_prompt = sum(o["prompt"] for o in outs)
    r = dict(ctx=ctx, conc=conc, prompt_tokens=[o["prompt"] for o in outs], completion=[o["completion"] for o in outs],
             ttft_s=[round(x, 2) for x in ttft], prefill_agg_tok_s=round(tot_prompt / (max(o["ts"][0] for o in outs) - t0), 1),
             decode_per_stream=[round(x, 2) for x in dec], decode_agg_overlap=round(agg, 2) if agg else None,
             overlap_s=round(hi - lo, 1),
             itl_p50_ms=[round(1e3 * sorted(b - a for a, b in zip(o["ts"], o["ts"][1:]))[len(o["ts"]) // 2], 1) for o in outs],
             itl_max_s=[round(max(b - a for a, b in zip(o["ts"], o["ts"][1:])), 2) for o in outs],
             stall_s=[round(sum(b - a for a, b in zip(o["ts"], o["ts"][1:]) if b - a > 0.5), 1) for o in outs],
             t0_abs=round(t0, 3), first_abs=[round(o["ts"][0], 3) for o in outs], last_abs=[round(o["ts"][-1], 3) for o in outs])
    print(json.dumps(r), flush=True); return r
cells = [(int(c.split("x")[0]), int(c.split("x")[1])) for c in sys.argv[1].split(",")]
seed = 1000
cell(1024, 1, 1)                                   # warm (not recorded)
cell(8192, 1, 2)                                   # warm 8k shape (Triton/inductor compiles), not recorded
for ctx, conc in cells:
    cell(ctx, conc, seed); seed += 100

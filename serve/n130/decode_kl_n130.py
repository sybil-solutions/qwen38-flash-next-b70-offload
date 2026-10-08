# Decode-path quality, paired on the same server: greedy-generate each panel prompt through the decode path (top-20
# logprobs per step), flush the radix cache, then teacher-force prompt + generated ids through the prefill path
# (staged / RAM-resident experts, no masking) and compare, per generated position:
#   KL(prefill || decode) on the decode top-20 support (both renormalised), top-1 agreement.
# Generation runs to EOS: max_new_tokens = context - prompt (the most the context allows; /generate defaults to 128).
# The teacher-forced scoring call uses max_new_tokens 1 (scoring probe, same as kit/tools/score_ref_panel.py).
import json, math, sys, urllib.request
URL = "http://127.0.0.1:30330"; CTX = 65536
def post(path, payload, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        b = r.read()
        try: return json.loads(b)
        except Exception: return b.decode()
panel = json.load(open(sys.argv[1])); out = sys.argv[2]
tot = dict(pos=0, kl=0.0, top1=0); per = []
for i, it in enumerate(panel):
    P = it["prompt_len"]; prompt = it["tokens"][:P]
    g = post("/generate", {"input_ids": prompt, "return_logprob": True, "top_logprobs_num": 20,
                           "sampling_params": {"temperature": 0, "max_new_tokens": CTX - P}})
    m = g["meta_info"]; gen = [t[1] for t in m["output_token_logprobs"]]; dtop = m["output_top_logprobs"]
    post("/flush_cache", {}) if False else urllib.request.urlopen(urllib.request.Request(URL + "/flush_cache", method="POST"), timeout=60).read()
    support = sorted({e[1] for row in dtop for e in row})
    t = post("/generate", {"input_ids": prompt + gen, "return_logprob": True, "logprob_start_len": P - 1, "top_logprobs_num": 20,
                           "token_ids_logprob": support, "sampling_params": {"temperature": 0, "max_new_tokens": 1}})
    tm = t["meta_info"]
    # input entry j+1 holds the distribution that predicted token P+j (see score_ref_panel.py); last = first output entry
    ptop = list(tm["input_top_logprobs"][1:]) + [tm["output_top_logprobs"][0]]
    pids = list(tm["input_token_ids_logprobs"][1:]) + [tm["output_token_ids_logprobs"][0]]
    s = dict(pos=0, kl=0.0, top1=0, kls=[])
    for j in range(len(gen)):
        d = {e[1]: e[0] for e in dtop[j]}; p = {e[1]: e[0] for e in pids[j]}
        ids = list(d.keys())
        pd = [math.exp(d[x]) for x in ids]; zd = sum(pd)
        pp = [math.exp(p[x]) if p.get(x) is not None else 1e-30 for x in ids]; zp = sum(pp)
        kl = sum((a / zp) * (math.log(max(a / zp, 1e-30)) - math.log(max(b / zd, 1e-30))) for a, b in zip(pp, pd))
        parg = max(ptop[j], key=lambda e: e[0])[1]
        s["pos"] += 1; s["kl"] += kl; s["top1"] += int(parg == gen[j]); s["kls"].append(kl)
    ks = sorted(s["kls"])
    rec = dict(i=i, gen_tokens=len(gen), finish=m.get("finish_reason"), top1=s["top1"] / max(1, s["pos"]), kl=s["kl"] / max(1, s["pos"]),
               kl_p99=ks[int(0.99 * (len(ks) - 1))] if ks else None, kl_max=ks[-1] if ks else None)
    per.append(rec); print(json.dumps(rec), flush=True)
    for k in ("pos", "kl", "top1"): tot[k] += s[k]
    urllib.request.urlopen(urllib.request.Request(URL + "/flush_cache", method="POST"), timeout=60).read()
res = dict(positions=tot["pos"], decode_vs_prefill_top1=tot["top1"] / tot["pos"], decode_vs_prefill_kl=tot["kl"] / tot["pos"], per_prompt=per)
print(json.dumps({k: v for k, v in res.items() if k != "per_prompt"}), flush=True)
json.dump(res, open(out, "w"), indent=1)

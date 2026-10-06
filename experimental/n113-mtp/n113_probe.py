# N113 probe: greedy chat completions (thinking off, natural completion, no output cap) on fixed prompts; records the
# full text (for greedy-equality checks between spec and non-spec boots), token counts, wall time and, with MTP on,
# SGLang's per-request spec_accept_length / spec_verify_ct from meta_info.
# usage: python3 n113_probe.py <out.jsonl> [port]
import json, sys, time, urllib.request
OUT = sys.argv[1]
PORT = sys.argv[2] if len(sys.argv) > 2 else "30260"
U = f"http://127.0.0.1:{PORT}/v1/chat/completions"
PROMPTS = [
    ("short", "What is the capital of France? Answer in one sentence."),
    ("code1", "Write a Python function that checks whether a number is prime, with a short docstring."),
    ("code2", "Write a C function that reverses a singly linked list in place. Include the struct definition and a short comment."),
    ("prose1", "Explain in about 300 words how a mixture-of-experts language model routes tokens to experts."),
    ("prose2", "Describe the water cycle for a ten-year-old in two short paragraphs."),
]


def chat(q):
    b = {"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": q}],
         "chat_template_kwargs": {"enable_thinking": False}, "return_meta_info": True}
    t = time.time()
    req = urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return time.time() - t, r


with open(OUT, "w") as f:
    for name, q in PROMPTS:
        dt, r = chat(q)
        u = r.get("usage", {})
        ch = r["choices"][0]
        mi = ch.get("meta_info") or r.get("meta_info") or {}
        rec = {"name": name, "s": round(dt, 2), "prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
               "tok_s_e2e": round((u.get("completion_tokens") or 0) / dt, 2),
               "accept_len": mi.get("spec_accept_length"), "verify_ct": mi.get("spec_verify_ct"),
               "finish": ch.get("finish_reason"), "meta_keys": sorted(mi.keys())[:30],
               "text": ch["message"].get("content") or ""}
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "text"} | {"head": rec["text"][:120].replace("\n", " | ")}), flush=True)

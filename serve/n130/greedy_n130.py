# Greedy decode panel: the 8 prompts of kit/reference/qwen3.8-flash-next-exl3-ref-panel.json through the chat API,
# temperature 0, thinking off, natural completion (no max_tokens). Compares with the exllamav3 reference completion.
import json, sys, time, urllib.request
U = "http://127.0.0.1:30330/v1/chat/completions"
panel = json.load(open(sys.argv[1])); out = sys.argv[2]; res = []
for i, it in enumerate(panel):
    b = {"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": it["prompt"]}],
         "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
    txt = r["choices"][0]["message"]["content"] or ""; ref = it["completion"]; u = r["usage"]
    n = 0
    while n < min(len(txt), len(ref)) and txt[n] == ref[n]: n += 1
    rec = dict(i=i, prompt_tokens=u["prompt_tokens"], ref_prompt_len=it["prompt_len"], completion_tokens=u["completion_tokens"],
               ref_completion_tokens=len(it["tokens"]) - it["prompt_len"], exact=txt == ref, common_prefix_chars=n, ref_chars=len(ref),
               finish=r["choices"][0].get("finish_reason"), s=round(time.time() - t, 1), text=txt)
    res.append(rec); print(json.dumps({k: v for k, v in rec.items() if k != "text"}), flush=True)
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(dict(exact=sum(r["exact"] for r in res), n=len(res), prefix_frac=round(sum(r["common_prefix_chars"] for r in res) / sum(r["ref_chars"] for r in res), 4))), flush=True)

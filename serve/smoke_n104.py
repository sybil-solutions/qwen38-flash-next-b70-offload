import json, time, urllib.request, sys
U = "http://127.0.0.1:30260/v1/chat/completions"
def chat(q, think=False):
    b = {"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": q}], "chat_template_kwargs": {"enable_thinking": think}}
    t = time.time()
    r = json.loads(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    dt = time.time() - t; u = r["usage"]
    return dt, u, r["choices"][0]["message"]["content"] or ""
out = []
for q in ["What is the capital of France? Answer in one sentence.",
          "Write a Python function that checks whether a number is prime, with a short docstring.",
          "Explain in about 300 words how a mixture-of-experts language model routes tokens to experts."]:
    dt, u, txt = chat(q)
    out.append({"q": q[:40], "s": round(dt, 1), "prompt": u["prompt_tokens"], "completion": u["completion_tokens"],
                "tok_s_e2e": round(u["completion_tokens"] / dt, 2), "text": txt[:300].replace("\n", " | ")})
    print(json.dumps(out[-1]), flush=True)

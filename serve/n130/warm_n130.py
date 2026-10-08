# Operational warm-up after server start (not scored): two chat requests, temperature 0, natural completion.
import json, time, urllib.request
U = "http://127.0.0.1:30330/v1/chat/completions"
for q in ["Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
          "Explain how a hash map handles collisions, with a short Python example."]:
    b = {"model": "flashnext", "temperature": 0, "messages": [{"role": "user", "content": q}], "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time(); r = json.loads(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
    print(json.dumps(dict(s=round(time.time() - t, 1), completion=r["usage"]["completion_tokens"], text=(r["choices"][0]["message"]["content"] or "")[:200].replace("\n", " | "))), flush=True)

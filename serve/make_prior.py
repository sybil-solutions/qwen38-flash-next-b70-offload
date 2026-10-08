# expert frequency prior for Qwen3.8-Flash-Next from the real K02 routing traces: keys = layer*512 + e, hottest first
import glob, json, numpy as np
c = np.zeros(48 * 512, dtype=np.int64)
for f in sorted(glob.glob("/traces/*.npz")):
    z = np.load(f)
    for k in ("decode_ids", "prefill_ids"):
        if k in z:
            a = z[k].astype(np.int64)                      # [T, 48, topk]
            for l in range(48):
                v = a[:, l].reshape(-1); v = v[v >= 0]
                c[l * 512:(l + 1) * 512] += np.bincount(v, minlength=512)
order = np.argsort(-c)
json.dump({"keys": order.tolist(), "counts": c[order].tolist()}, open("/out/qwen_expert_prior.json", "w"))
print(json.dumps({"nonzero": int((c > 0).sum()), "top8000_share": round(float(c[order[:8000]].sum() / c.sum()), 4)}))

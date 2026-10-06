"""N114 QSA: offline mock test of patch_qsa_dec.py (Mac / CPU, no SYCL, no SGLang install).

Fake SGLang modules reproduce the call structure the patch hooks (image-era sg2 indexer + exl3xpu's closure-style
select_decode_tokens and _l2p, D206-style backend _forward_paged_attention with its non-CUDA branch); the op is
qsa_dec.emulate. Checks: decode batches defer the expansion and take the fused op; speculative batches keep expanded
indices (fused, logical mode); every fallback (rows cap, kernel error) reproduces the unpatched output bit for bit;
nothing is patched without EXL3_QSA_DEC_SYCL=1.
usage: python3 mock_patch_qsa_dec.py
"""
from __future__ import annotations

import importlib
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["EXL3_QSA_DEC_SYCL"] = "1"
from offline_qsa_sem import SRC, load_exl3xpu, load_fn, scenario  # noqa: E402

EXPAND = load_fn(SRC["sg2"], "torch_expand_qsa_block_indices")
EX = load_exl3xpu()


def mkmod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


for pkg in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.attention",
            "sglang.srt.layers.attention.qsa", "sglang.srt.models"):
    mkmod(pkg)
    sys.modules[pkg].__path__ = []

# ---------------------------------------------------------------- fake indexer (sg2 structure, exl3xpu select)
SC = {}


class QSAIndexer:
    compress_ratio = 4
    token_topk = 2048
    block_topk = 512

    def forward_cuda(self, hidden_states, positions, forward_batch, indexer_metadata):
        q = torch.zeros(1, 4, 128) if hidden_states is None else hidden_states
        return self.select_decode_tokens(q, None, None, None, 0, indexer_metadata["qpos"],
                                         indexer_metadata["seqlen"])


def _exl3xpu_install(cls):
    from_kernel = EXPAND               # closure variable named like exl3xpu's import
    expand_qsa_block_indices = from_kernel

    def select_decode_tokens(self, q, compressed_cache, compressed_page_table, compressed_lengths, max_model_len,
                             query_positions, sequence_lengths):
        out = SC["blk"].clone()        # stands in for _ops().qsa_decode_select(...)
        return expand_qsa_block_indices(out, query_positions, sequence_lengths, compress_ratio=self.compress_ratio,
                                        token_topk=self.token_topk)
    cls.select_decode_tokens = select_decode_tokens


_exl3xpu_install(QSAIndexer)
mkmod("sglang.srt.layers.attention.qsa.qsa_indexer", QSAIndexer=QSAIndexer)


# ---------------------------------------------------------------- fake model layer
class Layer:
    layer_id = 3
    scaling = 1.0 / 16

    def __init__(self):
        self.indexer = QSAIndexer()
        self.bound = self.indexer.forward_cuda          # bound at construction, like a CustomOp dispatch

    def _compute_qsa_topk_indices(self, hidden_states, positions, forward_batch):
        return self.bound(hidden_states, positions, forward_batch, SC["imeta"])


mkmod("sglang.srt.models.qwen4_exp", Qwen4ExpAttentionDecoderLayer=Layer)
Layer.__module__ = "sglang.srt.models.qwen4_exp"


# ---------------------------------------------------------------- fake backend (D206 non-CUDA branch, exl3xpu)
class Pool:
    def get_key_buffer(self, i):
        return SC["k"]

    def get_value_buffer(self, i):
        return SC["v"]


def qsa_sparse_attention(q, k, v, slots, scale):
    return EX.qsa_sparse_attention_reference(q, k, v, slots, scale)


class QwenSparseAttnBackend:
    def __init__(self):
        self.token_to_kv_pool = Pool()

    def _resolve_metadata(self, fb):
        return SC["meta"]

    def _uses_block_indices(self, indices):
        return indices.shape[1] == 512

    def _expand_block_indices(self, indices, metadata):
        if not self._uses_block_indices(indices):
            return indices
        im = SC["imeta"]
        return EXPAND(indices, im["qpos"], im["seqlen"], 4, 2048)

    @staticmethod
    def _logical_to_physical(logical_indices, metadata):
        sequence_ids = metadata.token_to_batch_idx.long()
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(0, sequence_ids)
        valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
        safe = logical_indices.clamp(min=0, max=metadata.token_slot_table.shape[1] - 1).long()
        slots = metadata.token_slot_table[sequence_ids[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)

    def _forward_paged_attention(self, q, layer, forward_batch, topk_indices):
        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        metadata = self._resolve_metadata(forward_batch)
        topk_indices = self._expand_block_indices(topk_indices, metadata)
        slots = self._logical_to_physical(topk_indices, metadata)
        output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
        return output.reshape(q.shape[0], -1)


def _exl3xpu_l2p(c):
    orig_fn = c.__dict__["_logical_to_physical"].__func__

    def _l2p(self, logical_indices, metadata, _orig=orig_fn):
        rr = getattr(metadata, "row_req_pool_indices", None)
        rt = getattr(self, "req_to_token", None)
        if not getattr(metadata, "is_cuda_graph", False) or rr is None or rt is None:
            return _orig(logical_indices, metadata)
        sequence_ids = metadata.token_to_batch_idx.long()
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(0, sequence_ids)
        valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
        safe = logical_indices.clamp(min=0, max=rt.shape[1] - 1).long()
        req = rr.long().index_select(0, sequence_ids)
        slots = rt[req[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)
    c._logical_to_physical = _l2p


_exl3xpu_l2p(QwenSparseAttnBackend)
mkmod("sglang.srt.layers.attention.qwen_sparse_attn_backend", QwenSparseAttnBackend=QwenSparseAttnBackend)
QwenSparseAttnBackend.__module__ = "sglang.srt.layers.attention.qwen_sparse_attn_backend"


class FM:
    def __init__(self, decode):
        self.d = decode

    def is_decode(self):
        return self.d


class FB:
    def __init__(self, decode=True, spec=None):
        self.forward_mode = FM(decode)
        self.spec_info = spec


def setup(R, verify, edges, graph, g):
    sc = scenario(g, R, verify, edges)
    SC.clear()
    SC["blk"] = sc["blk"]
    SC["k"] = torch.randn(sc["NS"], 2, 256, generator=g).to(torch.float8_e4m3fn)
    SC["v"] = torch.randn(sc["NS"], 2, 256, generator=g).to(torch.float8_e4m3fn)
    SC["imeta"] = {"qpos": sc["qpos"], "seqlen": sc["seqlen"]}
    meta = types.SimpleNamespace(token_to_batch_idx=sc["t2b"], sequence_lengths=sc["rowlen"],
                                 token_slot_table=sc["rt"][sc["rr"].long()], is_cuda_graph=graph,
                                 row_req_pool_indices=sc["rr"] if graph else None)
    SC["meta"] = meta
    return (torch.randn(R, 24, 256, generator=g) * 0.6).to(torch.bfloat16), sc


def main():
    import patch_qsa_dec as P
    import qsa_dec
    qsa_dec.load = lambda: True
    qsa_dec.why_not = lambda *a, **k: None
    emu = qsa_dec.emulate
    calls = []

    def fake_op(*a, chunk=None, cpw=None, sg=None, fp8=None, target_wg=None, **k):
        calls.append(k.get("expand"))
        if SC.get("raise"):
            raise RuntimeError("injected kernel error")
        return emu(*a, **k)
    qsa_dec.decode_attention = fake_op
    P._DEVICES.add("cpu")
    # unpatched reference outputs first (the patch wraps in place)
    g = torch.Generator().manual_seed(5)
    layer = Layer()
    backend = QwenSparseAttnBackend()
    backend.req_to_token = None
    fails = 0
    cases = []
    for R, verify, edges, graph in ((1, False, False, True), (4, False, True, True), (8, True, False, True),
                                    (2, False, False, False)):
        q, sc = setup(R, verify, edges, graph, g)
        backend.req_to_token = sc["rt"] if graph else None
        idx = layer._compute_qsa_topk_indices(None, None, FB())
        want = backend._forward_paged_attention(q, layer, FB(), idx)
        cases.append((R, verify, edges, graph, q, sc, want, SC["k"], SC["v"]))
    assert P.install(), "install failed"
    assert getattr(QSAIndexer.select_decode_tokens, "_n114", False)
    assert getattr(QwenSparseAttnBackend._forward_paged_attention, "_n114", False)
    assert getattr(Layer._compute_qsa_topk_indices, "_n114", False)
    for R, verify, edges, graph, q, sc, want, kk, vv in cases:
        SC["blk"], SC["k"], SC["v"] = sc["blk"], kk, vv
        SC["imeta"] = {"qpos": sc["qpos"], "seqlen": sc["seqlen"]}
        SC["meta"] = types.SimpleNamespace(token_to_batch_idx=sc["t2b"], sequence_lengths=sc["rowlen"],
                                           token_slot_table=sc["rt"][sc["rr"].long()], is_cuda_graph=graph,
                                           row_req_pool_indices=sc["rr"] if graph else None)
        backend.req_to_token = sc["rt"] if graph else None
        tag = f"R={R} verify={verify} edges={edges} graph={graph}"
        # 1. decode: deferred block ids -> fused expand
        calls.clear()
        idx = layer._compute_qsa_topk_indices(None, None, FB())
        deferred = idx.shape[1] == 512 and getattr(idx, "_n114_qsa", None) is not None
        out = backend._forward_paged_attention(q, layer, FB(), idx)
        e1 = float((out.float() - want.float()).abs().max())
        ok1 = deferred and calls == [True] and e1 <= 3e-2
        # 2. speculative batch: expanded indices, fused logical mode
        calls.clear()
        idx2 = layer._compute_qsa_topk_indices(None, None, FB(spec=object()))
        out2 = backend._forward_paged_attention(q, layer, FB(spec=object()), idx2)
        ok2 = idx2.shape[1] == 2051 and calls == [False] and torch.equal(out2, out)
        # 3. rows cap -> previous path with the original expansion: bit-identical to unpatched
        P.MAX_ROWS = 0
        idx3 = layer._compute_qsa_topk_indices(None, None, FB())
        out3 = backend._forward_paged_attention(q, layer, FB(), idx3)
        P.MAX_ROWS = 64
        ok3 = torch.equal(out3, want)
        print(f"[{'PASS' if ok1 and ok2 and ok3 else 'FAIL'}] {tag}: defer {deferred} fused-expand vs unpatched max "
              f"{e1:.2e} | spec batch logical-mode {ok2} | rows-cap fallback bit-identical {ok3}")
        fails += not (ok1 and ok2 and ok3)
    # 4. kernel error -> disabled, previous path bit-identical, later calls stay on the previous path
    R, verify, edges, graph, q, sc, want, kk, vv = cases[0]
    SC.update(blk=sc["blk"], k=kk, v=vv, imeta={"qpos": sc["qpos"], "seqlen": sc["seqlen"]})
    SC["meta"] = types.SimpleNamespace(token_to_batch_idx=sc["t2b"], sequence_lengths=sc["rowlen"],
                                       token_slot_table=sc["rt"][sc["rr"].long()], is_cuda_graph=graph,
                                       row_req_pool_indices=sc["rr"])
    backend.req_to_token = sc["rt"]
    SC["raise"] = True
    idx = layer._compute_qsa_topk_indices(None, None, FB())
    out = backend._forward_paged_attention(q, layer, FB(), idx)
    SC["raise"] = False
    idx = layer._compute_qsa_topk_indices(None, None, FB())
    out_b = backend._forward_paged_attention(q, layer, FB(), idx)
    ok4 = torch.equal(out, want) and torch.equal(out_b, want) and P._state["broken"] and idx.shape[1] == 2051
    print(f"[{'PASS' if ok4 else 'FAIL'}] kernel error -> disabled, previous path bit-identical, no further defer")
    fails += not ok4
    print("STATS", dict(P.STATS))
    print("ALL PASS" if not fails else f"SOME FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

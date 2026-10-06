// N114 CPU check of the fused decode QSA kernels (no torch, no GPU): runs DecKernel + MergeKernel on the OpenCL CPU
// device inside image 24c872759256 and compares with a double-precision reference of the full decode semantics that
// SGLang's non-CUDA paged path implements today:
//   torch_expand_qsa_block_indices (block * ratio + offset, masked to < seq_len, + the ratio - 1 pending-tail tokens)
//   -> _logical_to_physical (table[row_of(seq), clamp(logical)], valid iff 0 <= logical < row_len[seq])
//   -> qsa_sparse_attention_reference (softmax over valid slots, empty row -> 0, GQA h / 12).
// Rows R in {1, 2, 4, 8}, KV fp8 e4m3fn (exact and via-fp16 decode) and bf16, chunk 64/128/256, sub-group 16/32,
// auto / one / all chunks per work-group, expand and pre-expanded (logical) indices, verify-style rows (several rows
// per sequence, per-row expansion bound), int32 and int64 index vectors, req_to_token-style and token_slot_table-style
// tables, edge rows (seq_len < 4, seq_len 1, pad rows with seq_len 0, invalid blocks, duplicate cells).
// Build + run (from n114-decode, in the image, no --device):  csrc/cpu_test.sh qsa
#define N114_NO_TORCH
#include "qsa_dec.sycl"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace n114q;

static float hbf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t hf2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static double e4m3_host(uint8_t u) {
    const int s = u >> 7, e = (u >> 3) & 15, m = u & 7;
    const double v = e == 0 ? m * std::ldexp(1.0, -9) : (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
    return s ? -v : v;
}
static int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) --q; return q; }

constexpr int HK = 2, HQ = HK * G, NSLOT = 6144, NT = 6, TW = 3200, RATIO = 4, TOPK = 2048, NB = TOPK / RATIO;

struct World {
    std::vector<uint8_t> k8, v8;      // [NSLOT, HK, HD] fp8 codes
    std::vector<uint16_t> kb, vb;     // [NSLOT, HK, HD] bf16
    std::vector<int32_t> table;       // [NT, TW] (req_to_token)
};

struct Batch {
    int R = 0;
    std::vector<uint16_t> q;          // [R, HQ, HD]
    std::vector<int32_t> blk;         // [R, NB]
    std::vector<int64_t> qpos, seqlen;       // per row
    std::vector<int64_t> t2b;                // per row
    std::vector<int64_t> rowlen, rreq;       // per sequence
};

// host expansion + l2p -> per-row physical slot lists (-1 masked), in cell order
static std::vector<std::vector<int>> host_cells(const World& w, const Batch& b, const std::vector<int32_t>* tab_override,
                                                int tab_rows_identity) {
    std::vector<std::vector<int>> out(b.R);
    for (int r = 0; r < b.R; ++r) {
        const int64_t seq = b.t2b[r];
        const int64_t tlen = b.rowlen[seq];
        const int64_t trow = tab_rows_identity ? seq : b.rreq[seq];
        const std::vector<int32_t>& tab = tab_override ? *tab_override : w.table;
        for (int c = 0; c < TOPK + RATIO - 1; ++c) {
            int64_t lg;
            if (c < TOPK) {
                const int64_t bl = b.blk[(size_t)r * NB + c / RATIO];
                lg = bl >= 0 ? bl * RATIO + c % RATIO : -1;
                if (!(lg >= 0 && lg < b.seqlen[r])) lg = -1;
            } else {
                const int64_t o = c - TOPK, vis = b.qpos[r] + 1, ts = fdiv(vis, RATIO) * RATIO, cnt = vis - ts;
                lg = (o < cnt && ts + o < b.seqlen[r]) ? ts + o : -1;
            }
            int s = -1;
            if (lg >= 0 && lg < tlen) s = tab[(size_t)trow * TW + std::min<int64_t>(lg, TW - 1)];
            out[r].push_back(s);
        }
    }
    return out;
}

// logical indices (torch layout: valid entries first, stable) for the pre-expanded mode
static std::vector<int32_t> host_logical(const Batch& b, bool dup) {
    const int T = TOPK + RATIO - 1;
    std::vector<int32_t> out((size_t)b.R * T, -1);
    for (int r = 0; r < b.R; ++r) {
        std::vector<int32_t> v, inv;
        for (int c = 0; c < T; ++c) {
            int64_t lg;
            if (c < TOPK) {
                const int64_t bl = b.blk[(size_t)r * NB + c / RATIO];
                lg = bl >= 0 ? bl * RATIO + c % RATIO : -1;
                if (!(lg >= 0 && lg < b.seqlen[r])) lg = -1;
            } else {
                const int64_t o = c - TOPK, vis = b.qpos[r] + 1, ts = fdiv(vis, RATIO) * RATIO, cnt = vis - ts;
                lg = (o < cnt && ts + o < b.seqlen[r]) ? ts + o : -1;
            }
            (lg >= 0 ? v : inv).push_back((int32_t)lg);
        }
        if (dup && !v.empty() && !inv.empty()) { v.push_back(v[0]); inv.pop_back(); }   // one duplicated cell
        size_t i = 0;
        for (int32_t x : v) out[(size_t)r * T + i++] = x;
        for (int32_t x : inv) out[(size_t)r * T + i++] = x;
    }
    return out;
}

static void reference(const World& w, const Batch& b, const std::vector<std::vector<int>>& cells, int kvm, double scale,
                      std::vector<double>& out) {
    out.assign((size_t)b.R * HQ * HD, 0.0);
    for (int r = 0; r < b.R; ++r) {
        for (int h = 0; h < HQ; ++h) {
            const int kvh = h / G;
            std::vector<double> sc;
            std::vector<int> sl;
            for (int s : cells[r]) {
                if (s < 0) continue;
                double d = 0;
                for (int i = 0; i < HD; ++i) {
                    const size_t e = ((size_t)s * HK + kvh) * HD + i;
                    const double kk = kvm == KV_BF16 ? hbf(w.kb[e]) : e4m3_host(w.k8[e]);
                    d += (double)hbf(b.q[((size_t)r * HQ + h) * HD + i]) * kk;
                }
                sc.push_back(d * scale);
                sl.push_back(s);
            }
            if (sc.empty()) continue;
            double mx = -1e300;
            for (double x : sc) mx = std::max(mx, x);
            double L = 0;
            for (double& x : sc) { x = std::exp(x - mx); L += x; }
            for (int i = 0; i < HD; ++i) {
                double acc = 0;
                for (size_t j = 0; j < sc.size(); ++j) {
                    const size_t e = ((size_t)sl[j] * HK + kvh) * HD + i;
                    acc += sc[j] * (kvm == KV_BF16 ? hbf(w.vb[e]) : e4m3_host(w.v8[e]));
                }
                out[((size_t)r * HQ + h) * HD + i] = acc / L;
            }
        }
    }
}

template <typename T> static T* dev(sycl::queue& q, const std::vector<T>& v) {
    T* p = sycl::malloc_shared<T>(std::max<size_t>(v.size(), 1), q);
    std::copy(v.begin(), v.end(), p);
    return p;
}

struct Cfg { int kvm, sg, ch, cpw; bool expand, i64, rreq_table, dup; };

static int run_case(sycl::queue& q, const World& w, const Batch& b, const Cfg& c, const std::string& name,
                    const std::vector<int32_t>& seq_table) {
    const double scale = 1.0 / 16.0;
    // index vectors (int32 or int64)
    auto ivec_of = [&](const std::vector<int64_t>& v, std::vector<void*>& owned) {
        IVec iv;
        iv.n = (int64_t)v.size();
        iv.st = 1;
        iv.is64 = c.i64;
        if (c.i64) { auto* p = dev(q, v); owned.push_back(p); iv.p = p; }
        else { std::vector<int32_t> v32(v.begin(), v.end()); auto* p = dev(q, v32); owned.push_back(p); iv.p = p; }
        return iv;
    };
    std::vector<void*> owned;
    DecArgs a{};
    auto* qd = dev(q, b.q); owned.push_back(qd);
    a.q = qd; a.qs0 = (int64_t)HQ * HD; a.qs1 = HD;
    if (c.kvm == KV_BF16) {
        auto* kd = dev(q, w.kb); auto* vd = dev(q, w.vb); owned.push_back(kd); owned.push_back(vd);
        a.k = kd; a.v = vd;
    } else {
        auto* kd = dev(q, w.k8); auto* vd = dev(q, w.v8); owned.push_back(kd); owned.push_back(vd);
        a.k = kd; a.v = vd;
    }
    a.ks0 = (int64_t)HK * HD; a.ks1 = HD; a.vs0 = a.ks0; a.vs1 = a.ks1;
    std::vector<int32_t> logical;
    if (c.expand) {
        auto* bd = dev(q, b.blk); owned.push_back(bd);
        a.idx = bd; a.is0 = NB; a.W = NB;
        a.expand = 1; a.ratio = RATIO; a.token_topk = TOPK; a.T = TOPK + RATIO - 1;
        a.qpos = ivec_of(b.qpos, owned);
        a.seqlen = ivec_of(b.seqlen, owned);
    } else {
        logical = host_logical(b, c.dup);
        auto* ld_ = dev(q, logical); owned.push_back(ld_);
        a.idx = ld_; a.W = TOPK + RATIO - 1; a.is0 = a.W;
        a.expand = 0; a.ratio = 1; a.token_topk = a.W; a.T = a.W;
    }
    a.t2b = ivec_of(b.t2b, owned);
    a.rowlen = ivec_of(b.rowlen, owned);
    if (c.rreq_table) {
        a.rreq = ivec_of(b.rreq, owned);
        auto* td = dev(q, w.table); owned.push_back(td);
        a.table = td; a.NT = NT;
    } else {   // token_slot_table style: one table row per sequence, row = sequence id
        auto* td = dev(q, seq_table); owned.push_back(td);
        a.table = td; a.NT = (int64_t)b.rowlen.size();
    }
    a.tab64 = 0; a.ts0 = TW; a.TW = TW;
    const float vun = c.kvm == KV_FP8H ? 256.0f : 1.0f;
    a.qscale = (float)scale * LOG2E * vun;
    a.vunscale = vun;
    uint16_t* od = sycl::malloc_shared<uint16_t>((size_t)b.R * HQ * HD, q);
    std::fill(od, od + (size_t)b.R * HQ * HD, (uint16_t)0x7FC0);   // NaN sentinel: every element must be written
    a.out = od;
    geometry(a, b.R, HK, c.ch, c.cpw, 160);
    const int64_t nf = scratch_floats(a, b.R, HK);
    float* scr = nf > 0 ? sycl::malloc_shared<float>((size_t)nf, q) : nullptr;
    if (!launch(q, a, b.R, HK, c.kvm, c.sg, c.ch, scr)) { std::printf("bad cfg\n"); std::exit(2); }
    q.wait();

    // reference
    std::vector<std::vector<int>> cells;
    if (c.expand) {
        cells = host_cells(w, b, c.rreq_table ? nullptr : &seq_table, c.rreq_table ? 0 : 1);
    } else {
        cells.resize(b.R);
        for (int r = 0; r < b.R; ++r) {
            const int64_t seq = b.t2b[r];
            for (int i = 0; i < a.W; ++i) {
                const int64_t lg = logical[(size_t)r * a.W + i];
                int s = -1;
                if (lg >= 0 && lg < b.rowlen[seq]) {
                    const int64_t trow = c.rreq_table ? b.rreq[seq] : seq;
                    const std::vector<int32_t>& tab = c.rreq_table ? w.table : seq_table;
                    s = tab[(size_t)trow * TW + std::min<int64_t>(lg, TW - 1)];
                }
                cells[r].push_back(s);
            }
        }
    }
    std::vector<double> ref;
    reference(w, b, cells, c.kvm == KV_BF16 ? KV_BF16 : KV_FP8, scale, ref);
    double max_abs = 0, num = 0, den = 0, worst = 0;   // worst: max |d| / (|ref| 2^-8 + 2e-3) (bf16 RNE <= 2^-9 |x|)
    bool finite = true, zeros_ok = true;
    for (int r = 0; r < b.R; ++r) {
        bool empty = true;
        for (int s : cells[r]) empty &= s < 0;
        for (int i = 0; i < HQ * HD; ++i) {
            const size_t e = (size_t)r * HQ * HD + i;
            const float x = hbf(od[e]);
            if (!std::isfinite(x)) finite = false;
            if (empty && od[e] != 0) zeros_ok = false;
            const double d = std::fabs(x - ref[e]);
            max_abs = std::max(max_abs, d);
            worst = std::max(worst, d / (std::fabs(ref[e]) * (1.0 / 256) + 2e-3));
            num += d * d; den += ref[e] * ref[e];
        }
    }
    const double rel = std::sqrt(num / std::max(den, 1e-300));
    const bool ok = finite && zeros_ok && worst <= 1.0 && rel <= 4e-3;
    std::printf("[%s] %-58s splits %2d x cpw %2d | max_abs %.3e rel %.3e tol-ratio %.2f finite %d zero-rows %d\n", ok ? "PASS" : "FAIL",
                name.c_str(), a.n_split, a.cpw, max_abs, rel, worst, (int)finite, (int)zeros_ok);
    for (void* p : owned) sycl::free(p, q);
    sycl::free(od, q);
    if (scr) sycl::free(scr, q);
    return ok ? 0 : 1;
}

// R rows; verify: rows grouped per sequence with increasing positions (per-row expansion bound)
static Batch make_batch(std::mt19937& rng, int R, bool verify, bool edges) {
    Batch b;
    b.R = R;
    std::normal_distribution<float> nd(0.0f, 0.6f);
    b.q.resize((size_t)R * HQ * HD);
    for (auto& x : b.q) x = hf2bf(nd(rng));
    std::vector<int64_t> lens = {3000, 2600, 3, 1, 0, 1500, 7, 2051};   // sequence lengths pool (edge cases)
    int nseq = verify ? std::max(1, R / 4) : R;
    for (int s = 0; s < nseq; ++s) {
        int64_t L = edges ? lens[s % lens.size()] : (int64_t)(2200 + (rng() % 900));
        b.rowlen.push_back(L);
        b.rreq.push_back((s * 5 + 1) % NT);
    }
    for (int r = 0; r < R; ++r) {
        const int s = verify ? std::min(nseq - 1, r / 4) : r;
        b.t2b.push_back(s);
        int64_t L = b.rowlen[s];
        int64_t sl = L;
        if (verify) sl = std::max<int64_t>(0, L - 3 + (r % 4));            // causal prefix per draft row
        b.seqlen.push_back(sl);
        b.qpos.push_back(sl - 1);
        // block selection: up to NB distinct complete blocks of the row's visible prefix, a few -1 holes
        const int64_t nblk = sl / RATIO;
        std::vector<int32_t> pool((size_t)nblk);
        std::iota(pool.begin(), pool.end(), 0);
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int j = 0; j < NB; ++j) {
            int32_t v = j < (int)pool.size() ? pool[j] : -1;
            if (v >= 0 && (rng() % 37) == 0) v = -1;                          // holes
            b.blk.push_back(v);
        }
        if (edges && r == 1 && nblk > 0) b.blk[(size_t)r * NB + 5] = (int32_t)(nblk + 3);   // block past seq_len -> masked
    }
    return b;
}

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    World w;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t NE = (size_t)NSLOT * HK * HD;
    w.k8.resize(NE); w.v8.resize(NE); w.kb.resize(NE); w.vb.resize(NE);
    for (size_t i = 0; i < NE; ++i) {
        auto code = [&]() -> uint8_t {   // e4m3fn, exponent 3..9 (|x| in [2^-4, 2^3)), some zeros / subnormals, no NaN
            const uint32_t r = rng();
            if (r % 50 == 0) return (uint8_t)((r >> 8) & 0x87);                // zero / subnormal
            const uint32_t e = 3 + (r >> 8) % 7, m = (r >> 12) & 7, s = (r >> 16) & 1;
            return (uint8_t)((s << 7) | (e << 3) | m);
        };
        w.k8[i] = code(); w.v8[i] = code();
        w.kb[i] = hf2bf(nd(rng)); w.vb[i] = hf2bf(nd(rng));
    }
    w.table.resize((size_t)NT * TW);
    for (int t = 0; t < NT; ++t) {
        std::vector<int32_t> perm(NSLOT);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int i = 0; i < TW; ++i) w.table[(size_t)t * TW + i] = perm[i];
    }
    int fails = 0;
    for (int R : {1, 2, 4, 8}) {
        for (int variant = 0; variant < 3; ++variant) {
            const bool verify = variant == 2 && R >= 4, edges = variant == 1;
            if (variant == 2 && R < 4) continue;
            Batch b = make_batch(rng, R, verify, edges);
            // token_slot_table-style table: row s = table row rreq[s]
            std::vector<int32_t> seq_table((size_t)b.rowlen.size() * TW);
            for (size_t s = 0; s < b.rowlen.size(); ++s)
                std::copy(w.table.begin() + (size_t)b.rreq[s] * TW, w.table.begin() + (size_t)(b.rreq[s] + 1) * TW,
                          seq_table.begin() + s * TW);
            std::vector<Cfg> cfgs = {
                {KV_FP8H, 16, 64, 0, true, false, true, false},
                {KV_FP8, 16, 64, 1, true, true, true, false},
                {KV_BF16, 16, 128, 0, true, false, false, false},
                {KV_FP8H, 32, 256, 0, true, true, true, false},
                {KV_FP8H, 16, 64, 1000, false, false, true, true},     // all chunks in one work-group, logical + dup
                {KV_FP8, 32, 128, 2, false, true, false, false},
            };
            const char* vname = verify ? "verify" : edges ? "edges" : "decode";
            for (const Cfg& c : cfgs) {
                const char* kn = c.kvm == KV_BF16 ? "bf16" : c.kvm == KV_FP8 ? "fp8" : "fp8h";
                char name[160];
                std::snprintf(name, sizeof name, "R%d %s kv=%s sg%d ch%d cpw%d %s %s %s%s", R, vname, kn, c.sg, c.ch, c.cpw,
                              c.expand ? "expand" : "logical", c.i64 ? "i64" : "i32",
                              c.rreq_table ? "req_to_token" : "slot_table", c.dup ? " dup" : "");
                fails += run_case(q, w, b, c, name, seq_table);
            }
        }
    }
    std::printf(fails ? "SOME FAIL (%d)\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}

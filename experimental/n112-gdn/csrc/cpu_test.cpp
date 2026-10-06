// N112 CPU check of the SYCL GDN kernels (no torch, no GPU): runs NormKernel + RecKernel on the OpenCL CPU device
// inside image 24c872759256 and compares with a double-precision token-serial reference of SGLang's
// chunk_gated_delta_rule semantics (l2-normalised q/k, g = log decay, beta, state [slots, H, V, K], slot -1 = zero
// state / no write-back, varlen cu_seqlens) plus chunk continuation (two calls with the state carried in the pool
// vs one call). Build + run (from n112-gdn, in the image, no --device):
//   source /opt/intel/oneapi/setvars.sh; icpx -fsycl -O2 -std=c++17 csrc/cpu_test.cpp -o build/cpu_test && build/cpu_test
#define N112_NO_TORCH
#include "gdn_sycl.sycl"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace n112;

static float hbf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t hf2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

constexpr int HK_ = 4, H_ = 12;            // small head counts (GQA 3, like 16 / 48)
constexpr int C_ = 2 * HK_ * KD + H_ * VD;  // mixed row width
constexpr int SLOTS = 4;

struct Data {
    int T;
    uint16_t* mixed;   // [T, C_] (q | k | v)
    float* g; float* beta;
    float* rn;
};

// reference: returns o[T][H][V] and updates pool (double copy)
static void reference(const Data& d, const std::vector<int>& cu, const std::vector<int>& idx, std::vector<double>& pool,
                      std::vector<double>& o) {
    const int T = d.T;
    o.assign((size_t)T * H_ * VD, 0.0);
    std::vector<double> S((size_t)H_ * VD * KD);
    for (size_t n = 0; n < idx.size(); ++n) {
        const int slot = idx[n];
        if (cu[n + 1] <= cu[n]) continue;
        for (size_t i = 0; i < S.size(); ++i) S[i] = slot >= 0 ? pool[(size_t)slot * S.size() + i] : 0.0;
        for (int t = cu[n]; t < cu[n + 1]; ++t) {
            const uint16_t* row = d.mixed + (size_t)t * C_;
            for (int h = 0; h < H_; ++h) {
                const int hk = h / (H_ / HK_);
                double q[KD], k[KD], qs = 0, ks = 0;
                for (int i = 0; i < KD; ++i) {
                    q[i] = hbf(row[hk * KD + i]); k[i] = hbf(row[HK_ * KD + hk * KD + i]);
                    qs += q[i] * q[i]; ks += k[i] * k[i];
                }
                qs = 1.0 / std::sqrt(qs + 1e-6); ks = 1.0 / std::sqrt(ks + 1e-6);
                for (int i = 0; i < KD; ++i) { q[i] *= qs; k[i] *= ks; }
                const double dec = std::exp((double)d.g[(size_t)t * H_ + h]);
                const double b = d.beta[(size_t)t * H_ + h];
                for (int v = 0; v < VD; ++v) {
                    double* s = &S[((size_t)h * VD + v) * KD];
                    double kv = 0;
                    for (int i = 0; i < KD; ++i) { s[i] *= dec; kv += s[i] * k[i]; }
                    const double vv = hbf(row[2 * HK_ * KD + h * VD + v]);
                    const double dl = b * (vv - kv);
                    double ov = 0;
                    for (int i = 0; i < KD; ++i) { s[i] += k[i] * dl; ov += s[i] * q[i]; }
                    o[((size_t)t * H_ + h) * VD + v] = ov / std::sqrt((double)KD);
                }
            }
        }
        if (slot >= 0) for (size_t i = 0; i < S.size(); ++i) pool[(size_t)slot * S.size() + i] = S[i];
    }
}

static void run_sycl(sycl::queue& q, const Data& d, int t0, int T, const std::vector<int>& cu, const std::vector<int>& idx,
                     float* pool, uint16_t* o, int rg, int cb) {
    int* dcu = sycl::malloc_shared<int>(cu.size(), q);
    int* didx = sycl::malloc_shared<int>(idx.size(), q);
    std::copy(cu.begin(), cu.end(), dcu);
    std::copy(idx.begin(), idx.end(), didx);
    const uint16_t* base = d.mixed + (size_t)t0 * C_;
    NormKernel nk{base, base + HK_ * KD, C_, KD, C_, KD, T, HK_, 1e-6f, d.rn};
    submit_norm(q, nk);
    RecArgs a{};
    a.q = base; a.qst = C_; a.qsh = KD;
    a.k = base + HK_ * KD; a.kst = C_; a.ksh = KD;
    a.v = base + 2 * HK_ * KD; a.vst = C_; a.vsh = VD;
    a.g = d.g + (size_t)t0 * H_; a.gst = H_;
    a.beta = d.beta + (size_t)t0 * H_; a.bst = H_;
    a.rn = d.rn; a.T = T;
    a.state = pool; a.sss = (int64_t)H_ * VD * KD;
    a.idx = didx; a.cu = dcu; a.o = o + (size_t)t0 * H_ * VD;
    a.H = H_; a.Hk = HK_; a.scale = 1.0f / std::sqrt((float)KD);
    if (!dispatch_cfg(rg, cb, false, false, q, a, (int64_t)idx.size())) { std::printf("bad cfg\n"); std::exit(2); }
    q.wait();
    sycl::free(dcu, q);
    sycl::free(didx, q);
}

static double rel(const std::vector<double>& ref, const float* x, size_t n, double* maxabs) {
    double num = 0, den = 0, m = 0;
    for (size_t i = 0; i < n; ++i) { const double e = x[i] - ref[i]; num += e * e; den += ref[i] * ref[i]; m = std::max(m, std::fabs(e)); }
    *maxabs = m;
    return std::sqrt(num / std::max(den, 1e-300));
}

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const int T = 600;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    Data d{};
    d.T = T;
    d.mixed = sycl::malloc_shared<uint16_t>((size_t)T * C_, q);
    d.g = sycl::malloc_shared<float>((size_t)T * H_, q);
    d.beta = sycl::malloc_shared<float>((size_t)T * H_, q);
    d.rn = sycl::malloc_shared<float>((size_t)2 * T * HK_, q);
    for (size_t i = 0; i < (size_t)T * C_; ++i) { float x = 0.7f * nd(rng); d.mixed[i] = hf2bf(x / (1.0f + std::exp(-x))); }
    for (int i = 0; i < T * H_; ++i) {
        const float a = nd(rng), sp = std::log1p(std::exp(a));
        d.g[i] = -std::exp(std::log(0.5f + 15.5f * std::uniform_real_distribution<float>(0, 1)(rng)) * 0.5f) * sp * 0.3f;
        d.beta[i] = 1.0f / (1.0f + std::exp(-nd(rng)));
    }
    const size_t SS = (size_t)H_ * VD * KD;
    std::vector<float> pool0(SLOTS * SS);
    for (auto& x : pool0) x = 0.3f * nd(rng);
    float* pool = sycl::malloc_shared<float>(SLOTS * SS, q);
    uint16_t* o = sycl::malloc_shared<uint16_t>((size_t)T * H_ * VD, q);
    int fails = 0;

    struct Case { const char* name; std::vector<int> cu, idx; int rg, cb; };
    std::vector<Case> cases = {
        {"single 600 slot2 (4,32)", {0, T}, {2}, 4, 32},
        {"varlen 6 seqs len1/empty/-1 (4,32)", {0, 37, 300, 301, 301, 590, 600}, {1, 3, -1, 0, 2, -1}, 4, 32},
        {"varlen (8,16)", {0, 100, 600}, {3, 1}, 8, 16},
        {"varlen (2,64)", {0, 100, 600}, {3, 1}, 2, 64},
        {"varlen (4,16)", {0, 250, 600}, {0, 2}, 4, 16},
    };
    for (auto& c : cases) {
        std::copy(pool0.begin(), pool0.end(), pool);
        std::fill(o, o + (size_t)T * H_ * VD, 0);
        run_sycl(q, d, 0, T, c.cu, c.idx, pool, o, c.rg, c.cb);
        std::vector<double> rp(pool0.begin(), pool0.end()), ro;
        reference(d, c.cu, c.idx, rp, ro);
        std::vector<float> of((size_t)T * H_ * VD);
        for (size_t i = 0; i < of.size(); ++i) of[i] = hbf(o[i]);
        double mo, ms;
        const double ro_rel = rel(ro, of.data(), of.size(), &mo);
        const double rs_rel = rel(rp, pool, SLOTS * SS, &ms);
        bool untouched = true;
        for (int s = 0; s < SLOTS; ++s) {
            bool used = false;
            for (size_t n = 0; n < c.idx.size(); ++n) used |= (c.idx[n] == s && c.cu[n + 1] > c.cu[n]);
            if (!used) untouched &= std::memcmp(pool + s * SS, pool0.data() + s * SS, SS * 4) == 0;
        }
        const bool ok = ro_rel < 4e-3 && rs_rel < 1e-5 && untouched;
        fails += !ok;
        std::printf("[%s] %-40s o rel %.3e max %.3e | state rel %.3e max %.3e | untouched %d\n", ok ? "PASS" : "FAIL",
                    c.name, ro_rel, mo, rs_rel, ms, (int)untouched);
    }
    // continuation: 600 tokens as 2 calls of 300 (state carried in the pool) vs one call
    {
        std::copy(pool0.begin(), pool0.end(), pool);
        run_sycl(q, d, 0, T, {0, T}, {1}, pool, o, 4, 32);
        std::vector<float> s1(pool + SS, pool + 2 * SS);
        std::vector<uint16_t> o1(o, o + (size_t)T * H_ * VD);
        std::copy(pool0.begin(), pool0.end(), pool);
        run_sycl(q, d, 0, 300, {0, 300}, {1}, pool, o, 4, 32);
        run_sycl(q, d, 300, 300, {0, 300}, {1}, pool, o, 4, 32);
        const bool same_s = std::memcmp(s1.data(), pool + SS, SS * 4) == 0;
        const bool same_o = std::memcmp(o1.data(), o, o1.size() * 2) == 0;
        fails += !(same_s && same_o);
        std::printf("[%s] continuation 2x300 vs 600: state bit-identical %d, o bit-identical %d\n",
                    same_s && same_o ? "PASS" : "FAIL", (int)same_s, (int)same_o);
    }
    std::printf(fails ? "SOME FAIL\n" : "ALL PASS\n");
    return fails ? 1 : 0;
}

// N114 CPU check of the SYCL GDN decode kernels (no torch, no GPU): runs ConvKernel + RecKernel on the OpenCL CPU
// device inside image 24c872759256 and compares with a reference of SGLang's decode semantics:
//   causal_conv1d_update (Triton: bf16 x bf16 products rounded to bf16, fp32 accumulation, silu, bf16 out, conv state
//   shift in a (slots, C, 3) view with the channel dim contiguous, pad slot -1 skipped) ->
//   fused_recurrent_gated_delta_rule_packed_decode (l2norm q/k, softplus/sigmoid gates, beta rounded to bf16, fp32
//   state, o stored bf16, pad slot -> 0) -> RMSNormGated(norm_before_gate, sigmoid|silu gate, bf16 out).
// The reference uses fp32 with the same rounding points (the "SGLang" result up to summation order) and a double
// variant for the error budget. Build + run (from n114-decode, in the image, no --device): csrc/cpu_test.sh gdn
#define N114_NO_TORCH
#include "gdn_dec.sycl"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace n114gdn;

static float hbf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t hf2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float hr(float f) { return hbf(hf2bf(f)); }

constexpr int HK_ = 4, H_ = 12;                    // GQA 3 like 16 / 48
constexpr int C_ = 2 * HK_ * KD + H_ * VD;          // conv channels (q | k | v)
constexpr int ZW = H_ * VD;
constexpr int QKVZ = C_ + ZW;                       // in_proj_qkvz row width
constexpr int D_ = 320;                             // hidden (ba fusion test)
constexpr int SLOTS = 6;
constexpr int W_ = 4;

struct Case { const char* name; std::vector<int> idx; int rg; bool sbf; int act; bool fuse_ba; bool idx64; };

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int BMAX = 8;
    // inputs (shared memory)
    uint16_t* qkvz = sycl::malloc_shared<uint16_t>((size_t)BMAX * QKVZ, q);
    uint16_t* ba = sycl::malloc_shared<uint16_t>((size_t)BMAX * 2 * H_, q);
    uint16_t* bx = sycl::malloc_shared<uint16_t>((size_t)BMAX * D_, q);
    uint16_t* wba = sycl::malloc_shared<uint16_t>((size_t)2 * H_ * D_, q);
    uint16_t* cw = sycl::malloc_shared<uint16_t>((size_t)C_ * W_, q);
    uint16_t* cst = sycl::malloc_shared<uint16_t>((size_t)SLOTS * 3 * C_, q);     // memory [slot][tok][C]
    float* ssm = sycl::malloc_shared<float>((size_t)SLOTS * H_ * VD * KD, q);
    uint16_t* ssm_bf = sycl::malloc_shared<uint16_t>((size_t)SLOTS * H_ * VD * KD, q);
    float* alog = sycl::malloc_shared<float>(H_, q);
    uint16_t* dtb = sycl::malloc_shared<uint16_t>(H_, q);   // bf16 dt_bias (tests the dtype path)
    uint16_t* nw = sycl::malloc_shared<uint16_t>(VD, q);
    int32_t* idx32 = sycl::malloc_shared<int32_t>(BMAX, q);
    int64_t* idx64 = sycl::malloc_shared<int64_t>(BMAX, q);
    uint16_t* conv_out = sycl::malloc_shared<uint16_t>((size_t)BMAX * C_, q);
    uint16_t* y = sycl::malloc_shared<uint16_t>((size_t)BMAX * H_ * VD, q);

    for (size_t i = 0; i < (size_t)BMAX * QKVZ; ++i) qkvz[i] = hf2bf(nd(rng));
    for (size_t i = 0; i < (size_t)BMAX * 2 * H_; ++i) ba[i] = hf2bf(1.5f * nd(rng));
    for (size_t i = 0; i < (size_t)BMAX * D_; ++i) bx[i] = hf2bf(nd(rng));
    for (size_t i = 0; i < (size_t)2 * H_ * D_; ++i) wba[i] = hf2bf(0.08f * nd(rng));
    for (size_t i = 0; i < (size_t)C_ * W_; ++i) cw[i] = hf2bf(0.5f * nd(rng));
    std::vector<uint16_t> cst0((size_t)SLOTS * 3 * C_);
    for (auto& v : cst0) v = hf2bf(nd(rng));
    std::vector<float> ssm0((size_t)SLOTS * H_ * VD * KD);
    for (auto& v : ssm0) v = 0.05f * nd(rng);
    for (int h = 0; h < H_; ++h) { alog[h] = std::log(1.0f + 15.0f * std::uniform_real_distribution<float>(0, 1)(rng)); dtb[h] = hf2bf(nd(rng)); }
    for (int i = 0; i < VD; ++i) nw[i] = hf2bf(1.0f + 0.2f * nd(rng));
    const float scale = 1.0f / std::sqrt((float)KD), eps = 1e-6f;

    std::vector<Case> cases = {
        {"B1 slot3 rg4 fp32 sigmoid", {3}, 4, false, 0, false, false},
        {"B2 rg4 fp32 sigmoid", {1, 4}, 4, false, 0, false, false},
        {"B4 pad -1 rg4 fp32 sigmoid i64", {5, -1, 0, 2}, 4, false, 0, false, true},
        {"B4 rg8 fp32 sigmoid", {0, 1, 2, 3}, 8, false, 0, false, false},
        {"B4 rg2 fp32 silu", {3, 2, 1, 0}, 2, false, 1, false, false},
        {"B3 rg4 bf16-state sigmoid", {2, 0, 5}, 4, true, 0, false, false},
        {"B4 rg4 fp32 act=none (rec only)", {0, 1, 2, 3}, 4, false, 2, false, false},
        {"B4 rg4 fp32 sigmoid ba-fused", {4, 0, -1, 1}, 4, false, 0, true, false},
        {"B8 rg4 fp32 sigmoid", {0, 1, 2, 3, 4, 5, -1, -1}, 4, false, 0, false, false},
    };
    int fails = 0;
    for (const Case& c : cases) {
        const int B = (int)c.idx.size();
        for (int i = 0; i < B; ++i) { idx32[i] = c.idx[i]; idx64[i] = c.idx[i]; }
        std::copy(cst0.begin(), cst0.end(), cst);
        std::copy(ssm0.begin(), ssm0.end(), ssm);
        for (size_t i = 0; i < ssm0.size(); ++i) ssm_bf[i] = hf2bf(ssm0[i]);
        std::fill(y, y + (size_t)B * H_ * VD, (uint16_t)0x7fc0);
        // ---- device
        ConvArgs ca{};
        ca.x = qkvz; ca.xs = QKVZ; ca.w = cw; ca.ws0 = W_; ca.ws1 = 1;
        ca.st = cst; ca.ss0 = 3 * C_; ca.ss1 = 1; ca.ss2 = C_; ca.lines = SLOTS;
        ca.idx = c.idx64 ? (const void*)idx64 : (const void*)idx32; ca.idx64 = c.idx64;
        ca.out = conv_out; ca.B = B; ca.C = C_; ca.silu = true; ca.round_prod = true;
        dispatch_conv(W_, q, ca);
        RecArgs ra{};
        ra.qkv = conv_out; ra.qs = C_;
        if (c.fuse_ba) { ra.bx = bx; ra.bxs = D_; ra.D = D_; ra.bw = wba; }
        else { ra.bv = ba; ra.bs_ = 2 * H_; ra.av = ba + H_; ra.as_ = 2 * H_; }
        ra.alog = alog; ra.alog_dt = 0; ra.dtb = dtb; ra.dtb_dt = 1;
        ra.state = c.sbf ? (void*)ssm_bf : (void*)ssm; ra.sss = (int64_t)H_ * VD * KD; ra.lines = SLOTS;
        ra.idx = ca.idx; ra.idx64 = c.idx64;
        ra.z = qkvz + C_; ra.zs = QKVZ;
        ra.nw = nw; ra.nw_dt = 1; ra.eps = eps; ra.act = c.act; ra.scale = scale; ra.softplus_thr = 20.0f;
        ra.y = y; ra.H = H_; ra.Hk = HK_;
        dispatch_rec(c.rg, c.sbf, false, q, ra, B);
        q.wait();

        // ---- reference (fp32 with SGLang rounding points; double for the recurrence error budget)
        std::vector<uint16_t> rcst = cst0;
        std::vector<double> rs(ssm0.begin(), ssm0.end());
        if (c.sbf) for (auto& v : rs) v = hr((float)v);
        int conv_bad = 0, conv_n = 0, cst_bad = 0;
        double ye = 0, yd = 0, ymax = 0, se = 0, sd = 0;
        int y_eq = 0, y_n = 0, untouched_ok = 1;
        for (int r = 0; r < B; ++r) {
            const int slot = c.idx[r];
            if (slot < 0) {
                for (int i = 0; i < H_ * VD; ++i) { y_eq += y[(size_t)r * H_ * VD + i] == 0; ++y_n; }
                continue;
            }
            std::vector<uint16_t> co(C_);
            for (int ch = 0; ch < C_; ++ch) {
                uint16_t win[W_];
                for (int j = 0; j < 3; ++j) win[j] = rcst[(size_t)slot * 3 * C_ + (size_t)j * C_ + ch];
                win[3] = qkvz[(size_t)r * QKVZ + ch];
                float acc = 0.0f;
                for (int j = 0; j < W_; ++j) acc += hr(hbf(win[j]) * hbf(cw[ch * W_ + j]));
                for (int j = 0; j < 3; ++j) rcst[(size_t)slot * 3 * C_ + (size_t)j * C_ + ch] = win[j + 1];
                acc = acc / (1.0f + std::exp(-acc));
                co[ch] = hf2bf(acc);
                conv_bad += co[ch] != conv_out[(size_t)r * C_ + ch]; ++conv_n;
            }
            for (int h = 0; h < H_; ++h) {
                const int hk = h / (H_ / HK_);
                double qv[KD], kv_[KD], sq = 0, sk = 0;
                for (int i = 0; i < KD; ++i) {
                    qv[i] = hbf(co[hk * KD + i]); kv_[i] = hbf(co[(HK_ + hk) * KD + i]);
                    sq += qv[i] * qv[i]; sk += kv_[i] * kv_[i];
                }
                for (int i = 0; i < KD; ++i) { qv[i] = qv[i] / std::sqrt(sq + 1e-6) * scale; kv_[i] /= std::sqrt(sk + 1e-6); }
                float av, bv;
                if (c.fuse_ba) {
                    double pa = 0, pb = 0;
                    for (int i = 0; i < D_; ++i) {
                        pb += (double)hbf(bx[(size_t)r * D_ + i]) * hbf(wba[(size_t)h * D_ + i]);
                        pa += (double)hbf(bx[(size_t)r * D_ + i]) * hbf(wba[(size_t)(H_ + h) * D_ + i]);
                    }
                    av = hr((float)pa); bv = hr((float)pb);
                } else {
                    bv = hbf(ba[(size_t)r * 2 * H_ + h]); av = hbf(ba[(size_t)r * 2 * H_ + H_ + h]);
                }
                const double xg = (double)av + hbf(dtb[h]);
                const double sp = xg <= 20.0 ? std::log(1.0 + std::exp(xg)) : xg;
                const double g = -std::exp((double)alog[h]) * sp;
                const double beta = hr((float)(1.0 / (1.0 + std::exp(-(double)bv))));
                const double dec = std::exp(g);
                double o[VD];
                for (int v = 0; v < VD; ++v) {
                    double* s = &rs[(((size_t)slot * H_ + h) * VD + v) * KD];
                    double kvd = 0;
                    for (int i = 0; i < KD; ++i) { s[i] *= dec; kvd += s[i] * kv_[i]; }
                    const double d = (hbf(co[2 * HK_ * KD + h * VD + v]) - kvd) * beta;
                    double ov = 0;
                    for (int i = 0; i < KD; ++i) { s[i] += kv_[i] * d; ov += s[i] * qv[i]; }
                    o[v] = hr((float)ov);
                }
                double ss = 0;
                for (int v = 0; v < VD; ++v) ss += o[v] * o[v];
                const double rstd = 1.0 / std::sqrt(ss / VD + eps);
                for (int v = 0; v < VD; ++v) {
                    double ref;
                    if (c.act == 2) ref = o[v];
                    else {
                        const double z = hbf(qkvz[(size_t)r * QKVZ + C_ + h * VD + v]);
                        const double sg = 1.0 / (1.0 + std::exp(-z));
                        ref = (o[v] * rstd) * hbf(nw[v]) * (c.act == 1 ? z * sg : sg);
                    }
                    const double got = hbf(y[(size_t)r * H_ * VD + h * VD + v]);
                    ye += (got - ref) * (got - ref); yd += ref * ref; ymax = std::max(ymax, std::fabs(got - ref));
                    y_eq += hf2bf((float)ref) == y[(size_t)r * H_ * VD + h * VD + v]; ++y_n;
                }
            }
        }
        // conv state + ssm state compare (all slots: untouched slots must be bit-identical)
        for (size_t i = 0; i < rcst.size(); ++i) cst_bad += rcst[i] != cst[i];
        for (size_t i = 0; i < rs.size(); ++i) {
            const double got = c.sbf ? (double)hbf(ssm_bf[i]) : (double)ssm[i];
            se += (got - rs[i]) * (got - rs[i]); sd += rs[i] * rs[i];
        }
        for (int s = 0; s < SLOTS; ++s) {
            if (std::find(c.idx.begin(), c.idx.end(), s) != c.idx.end()) continue;
            const size_t n = (size_t)H_ * VD * KD;
            if (c.sbf) { for (size_t i = 0; i < n; ++i) untouched_ok &= ssm_bf[s * n + i] == hf2bf(ssm0[s * n + i]); }
            else untouched_ok &= std::memcmp(ssm + s * n, ssm0.data() + s * n, n * 4) == 0;
        }
        const double yrel = std::sqrt(ye / std::max(yd, 1e-300)), srel = std::sqrt(se / std::max(sd, 1e-300));
        const double eq = (double)y_eq / std::max(1, y_n);
        const double stol = c.sbf ? 4e-3 : 2e-6;
        const bool ok = conv_bad * 1000 <= conv_n && cst_bad == 0 && yrel < 6e-3 && srel < stol && eq > 0.95 && untouched_ok;
        fails += !ok;
        std::printf("[%s] %-36s conv mism %d/%d convstate mism %d | y rel %.2e max %.2e bf16-eq %.4f | state rel %.2e | untouched %d\n",
                    ok ? "PASS" : "FAIL", c.name, conv_bad, conv_n, cst_bad, yrel, ymax, eq, srel, untouched_ok);
    }
    std::printf(fails ? "SOME FAIL\n" : "ALL PASS\n");
    return fails ? 1 : 0;
}

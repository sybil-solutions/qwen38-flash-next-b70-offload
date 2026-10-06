// N114-hc CPU check of the SYCL hyper-connection decode kernels (no torch, no GPU): runs pre / down / up / combine on
// the OpenCL CPU device inside image 24c872759256 and compares with a host reference of SGLang's GatedResidual with
// its bf16 rounding points (norm -> n, t, a = silu(t/hc), u, mix, l, g, combine). Two references per case:
//   ref  : everything from x (double dots, rounded at the same points)  -> n is checked against it (<= 1 bf16 ulp)
//   ref2 : the downstream ops recomputed from the KERNEL's n             -> t/a/u/mix/l are checked against it, so a
//          1-ulp n difference (fp32 sum order) does not hide or fake a downstream error
//   fold : the folded combine R' = bf16(R + bo * g) must be bit-identical to the reference (pure elementwise)
// Build + run (from n114-decode, in the image, no --device):  csrc/cpu_test.sh hc
#define N114_NO_TORCH
#include "hc_dec.sycl"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace n114hc;

static float hbf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t hf2bf(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16);
}
static float hr(float f) { return hbf(hf2bf(f)); }
static float hsig(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static int ulp(uint16_t a, uint16_t b) {   // distance in bf16 steps (same sign assumed for small diffs)
    auto key = [](uint16_t v) { return (v & 0x8000) ? -(int)(v & 0x7fff) : (int)(v & 0x7fff); };
    return std::abs(key(a) - key(b));
}

struct Case { const char* name; int M, hs, hc, LR, KS, UC, MODE; bool fold, wi; int wdt; int xpad; };  // wdt 0 fp32 1 bf16 2 fp16

struct Stat { size_t n = 0, eq = 0; int maxu = 0; void add(uint16_t a, uint16_t b) { ++n; eq += a == b; maxu = std::max(maxu, ulp(a, b)); }
              double frac() const { return n ? (double)eq / n : 1.0; } };

static int run_case(sycl::queue& q, const Case& c, std::mt19937& rng) {
    const int M = c.M, hs = c.hs, hc = c.hc, D = hs * hc, LR = c.LR, nwi = c.wi ? hc : 0, R = LR + nwi;
    const int xs = D + c.xpad;
    if (!plan_ok(M, hs, hc, LR, c.KS, c.UC)) { std::printf("[FAIL] %s: plan rejected\n", c.name); return 1; }
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto bfv = [&](size_t n, float s) { std::vector<uint16_t> v(n); for (auto& e : v) e = hf2bf(s * nd(rng)); return v; };
    std::vector<uint16_t> hx = bfv((size_t)M * xs, 1.5f), hbo = bfv((size_t)M * hs, 0.8f);
    std::vector<uint16_t> hwd = bfv((size_t)LR * D, 0.02f), hwu = bfv((size_t)D * LR, 0.06f), hwi = bfv((size_t)hc * D, 0.02f);
    std::vector<float> hw(D), hl((size_t)M * hc);
    for (auto& e : hw) e = c.wdt == 2 ? (float)(_Float16)(0.2f * nd(rng)) : hr(0.2f * nd(rng));
    for (auto& e : hl) e = hr(3.0f * nd(rng));

    auto sh16 = [&](const std::vector<uint16_t>& v) { auto* p = sycl::malloc_shared<uint16_t>(std::max<size_t>(v.size(), 8), q); std::copy(v.begin(), v.end(), p); return p; };
    uint16_t *dx = sh16(hx), *dbo = sh16(hbo), *dwd = sh16(hwd), *dwu = sh16(hwu), *dwi = sh16(hwi);
    void* dw;
    if (c.wdt == 0) { float* p = sycl::malloc_shared<float>(D, q); std::copy(hw.begin(), hw.end(), p); dw = p; }
    else if (c.wdt == 2) { uint16_t* p = sycl::malloc_shared<uint16_t>(D, q);
        for (int i = 0; i < D; ++i) { _Float16 h = (_Float16)hw[i]; std::memcpy(&p[i], &h, 2); } dw = p; }
    else { uint16_t* p = sycl::malloc_shared<uint16_t>(D, q); for (int i = 0; i < D; ++i) p[i] = hf2bf(hw[i]); dw = p; }
    float* dl = sycl::malloc_shared<float>(M * hc, q); std::copy(hl.begin(), hl.end(), dl);
    uint16_t* drout = sycl::malloc_shared<uint16_t>((size_t)M * D, q);
    uint16_t* dn = sycl::malloc_shared<uint16_t>((size_t)M * D, q);
    float* dpart = sycl::malloc_shared<float>((size_t)c.KS * M * R, q);
    uint16_t* dmix = sycl::malloc_shared<uint16_t>((size_t)M * hs, q);
    float* dlo = sycl::malloc_shared<float>(M * hc, q);
    uint16_t* dcomb = sycl::malloc_shared<uint16_t>((size_t)M * D, q);
    std::fill(dlo, dlo + M * hc, -12345.0f);

    MixPlan p{M, hs, hc, LR, c.KS, c.UC, c.MODE, c.fold, c.wdt, 1e-6f};
    run_mix(q, p, dx, xs, dbo, hs, dl, drout, dw, dwd, dwu, c.wi ? dwi : nullptr, nwi, dn, dpart, dmix, hs, dlo);
    // non-folded combine of the result (R = the residual this half normalised, bo, l of this half)
    if (c.wi) {
        CombArgs ca{c.fold ? drout : dx, c.fold ? (int64_t)D : (int64_t)xs, dbo, hs, dlo, dcomb, hs, hc};
        submit_combine(q, ca, M);
    }
    q.wait();

    // ---- reference
    std::vector<uint16_t> rres((size_t)M * D);     // the residual the norm sees
    for (int m = 0; m < M; ++m)
        for (int col = 0; col < D; ++col) {
            const float r = hbf(hx[(size_t)m * xs + col]);
            if (c.fold) {
                const int j = col / hs, d = col % hs;
                const float g = hr(2.0f * hsig(hl[m * hc + j] / (float)hc));
                volatile float pr = hbf(hbo[(size_t)m * hs + d]) * g;
                volatile float s = r + pr;
                rres[(size_t)m * D + col] = hf2bf(s);
            } else rres[(size_t)m * D + col] = hx[(size_t)m * xs + col];
        }
    Stat sfold, sn, smix, sl, scomb;
    double tmax = 0;   // max relative error of the fp32 dots (vs sum |terms|)
    if (c.fold) for (size_t i = 0; i < rres.size(); ++i) sfold.add(drout[i], rres[i]);
    std::vector<uint16_t> rn((size_t)M * D);
    for (int m = 0; m < M; ++m)
        for (int j = 0; j < hc; ++j) {
            double ss = 0;
            for (int d = 0; d < hs; ++d) { const double v = hbf(rres[(size_t)m * D + j * hs + d]); ss += v * v; }
            const float rs = (float)(1.0 / std::sqrt(ss / hs + 1e-6));
            for (int d = 0; d < hs; ++d) {
                const int col = j * hs + d;
                rn[(size_t)m * D + col] = hf2bf((hbf(rres[(size_t)m * D + col]) * rs) * (1.0f + hw[col]));
            }
        }
    for (size_t i = 0; i < rn.size(); ++i) sn.add(dn[i], rn[i]);
    // ref2 from the kernel's n
    for (int m = 0; m < M; ++m) {
        std::vector<float> a(LR);
        for (int k = 0; k < LR; ++k) {
            double t = 0, ta = 0;
            for (int d = 0; d < D; ++d) {
                const double p = (double)hbf(dn[(size_t)m * D + d]) * hbf(hwd[(size_t)k * D + d]);
                t += p; ta += std::fabs(p);
            }
            // the kernel's fp32 split-K sum (fixed split order), checked against the exact dot; the downstream
            // reference continues from the kernel's t so that a bf16 rounding tie of t (fp32 order) is not counted
            float tk = 0.0f;
            for (int s = 0; s < c.KS; ++s) tk += dpart[((size_t)s * M + m) * R + k];
            const double terr = std::fabs((double)tk - t) / (ta + 1e-30);
            tmax = std::max(tmax, terr);
            const float x = hr(tk) / (float)hc;
            a[k] = hr(x * hsig(x));
        }
        for (int d = 0; d < hs; ++d) {
            float s = 0;
            for (int j = 0; j < hc; ++j) {
                const int row = j * hs + d;
                double u = 0;
                for (int k = 0; k < LR; ++k) u += (double)a[k] * hbf(hwu[(size_t)row * LR + k]);
                s += hsig(hr((float)u)) * hbf(dn[(size_t)m * D + row]);
            }
            smix.add(dmix[(size_t)m * hs + d], hf2bf(s / (float)hc));
        }
        if (c.wi) {
            std::vector<float> lv(hc);
            for (int j = 0; j < hc; ++j) {
                double t = 0;
                double ta = 0;
                for (int d = 0; d < D; ++d) {
                    const double p = (double)hbf(dn[(size_t)m * D + d]) * hbf(hwi[(size_t)j * D + d]);
                    t += p; ta += std::fabs(p);
                }
                float tk = 0.0f;
                for (int s = 0; s < c.KS; ++s) tk += dpart[((size_t)s * M + m) * R + LR + j];
                tmax = std::max(tmax, std::fabs((double)tk - t) / (ta + 1e-30));
                lv[j] = hr(tk);
                sl.add(hf2bf(dlo[m * hc + j]), hf2bf(lv[j]));
            }
            // combine from the kernel's l (pure elementwise: must be exact)
            for (int col = 0; col < D; ++col) {
                const int j = col / hs, d = col % hs;
                const float g = hr(2.0f * hsig(dlo[m * hc + j] / (float)hc));
                volatile float pr = hbf(hbo[(size_t)m * hs + d]) * g;
                volatile float s = hbf(rres[(size_t)m * D + col]) + pr;
                scomb.add(dcomb[(size_t)m * D + col], hf2bf(s));
            }
        }
    }
    const bool ok = (!c.fold || sfold.eq == sfold.n) && sn.maxu <= 1 && sn.frac() >= 0.999 && smix.maxu <= 2 &&  // OpenCL CPU exp differs from std::exp
                   
                    smix.frac() >= 0.999 && tmax < 1e-5 && (!c.wi || (sl.maxu <= 1 && scomb.eq == scomb.n));
    std::printf("[%s] %-34s fold %s n eq %.5f maxulp %d | mix eq %.5f maxulp %d | l eq %.3f maxulp %d | combine eq %.5f | dot relerr %.1e\n",
                ok ? "PASS" : "FAIL", c.name, c.fold ? (sfold.eq == sfold.n ? "exact" : "DIFF") : "-", sn.frac(), sn.maxu,
                smix.frac(), smix.maxu, sl.frac(), sl.maxu, scomb.frac(), tmax);
    for (void* ptr : std::initializer_list<void*>{dx, dbo, dwd, dwu, dwi, dw, dl, drout, dn, dpart, dmix, dlo, dcomb}) sycl::free(ptr, q);
    return ok ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    std::vector<Case> cases = {
        // small geometry (fast): hs 256, hc 4, lowrank 64
        {"small M1 ks4 uc16 lane", 1, 256, 4, 64, 4, 16, 0, false, true, 1, 0},
        {"small M2 ks2 uc32 sg fold", 2, 256, 4, 64, 2, 32, 1, true, true, 1, 8},
        {"small M3 ks1 uc64 lane fold f32w", 3, 256, 4, 64, 1, 64, 0, true, true, 0, 0},
        {"small M4 ks8 uc16 sg", 4, 256, 4, 64, 8, 16, 1, false, true, 1, 16},
        {"small M8 ks4 uc32 lane fold", 8, 256, 4, 64, 4, 32, 0, true, true, 1, 0},
        {"small M5 ks4 noinject (mixer)", 5, 256, 4, 64, 4, 32, 0, false, false, 1, 0},
        {"small hc5 M2 ks4 uc16 (mtp hc+1)", 2, 256, 5, 64, 4, 16, 0, true, true, 1, 0},
        // real geometry: hs 2560, hc 4, lowrank 320
        {"real M1 ks8 uc32 lane", 1, 2560, 4, 320, 8, 32, 0, false, true, 1, 0},
        {"real M1 ks4 uc32 lane fold", 1, 2560, 4, 320, 4, 32, 0, true, true, 1, 0},
        {"real M2 ks8 uc32 sg fold", 2, 2560, 4, 320, 8, 32, 1, true, true, 1, 0},
        {"real M4 ks4 uc16 lane", 4, 2560, 4, 320, 4, 16, 0, false, true, 1, 0},
        {"real M8 ks4 uc64 sg fold", 8, 2560, 4, 320, 4, 64, 1, true, true, 0, 0},
        {"small M2 fold fp16 norm w", 2, 256, 4, 64, 4, 32, 0, true, true, 2, 0},
        {"real M1 ks8 fp16 norm w", 1, 2560, 4, 320, 8, 32, 0, false, true, 2, 0},
    };
    int fails = 0;
    for (auto& c : cases) fails += run_case(q, c, rng);
    std::printf(fails ? "SOME FAIL\n" : "ALL PASS\n");
    return fails ? 1 : 0;
}

// N108 CPU check of the SYCL kernels (no torch, no GPU): runs StrataKernel / BcastKernel (+ merge) on the OpenCL CPU
// device inside image 24c872759256 and compares with a double-precision reference of SGLang
// qsa_sparse_attention_reference semantics. Build + run (from n107-qsa, in the image):
//   source /opt/intel/oneapi/setvars.sh; icpx -fsycl -O2 -std=c++17 csrc/cpu_test.cpp -o build/cpu_test && build/cpu_test
#define N108_NO_TORCH
#include "qsa_row_sycl.sycl"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace n108;

static float host_e4m3(uint8_t u) {
    int s = u >> 7, e = (u >> 3) & 15, m = u & 7;
    float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp((float)m, -9);
    return s ? -v : v;
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) {   // RNE
    uint32_t u; std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

struct Case { int R, Hk, T, N; bool bf16kv; };

static int run_case(sycl::queue& q, const Case& c, int kernel, int cpw, int sg, int fp8_mode, unsigned seed) {
    const int R = c.R, Hk = c.Hk, Hq = G * Hk, T = c.T, N = c.N;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> qh((size_t)R * Hq * HD);
    for (auto& x : qh) x = f2bf(nd(rng));
    const size_t kvn = (size_t)N * Hk * HD;
    std::vector<uint8_t> k8(kvn), v8(kvn);
    std::vector<uint16_t> k16(kvn), v16(kvn);
    for (size_t i = 0; i < kvn; ++i) {
        uint8_t a, b;
        do { a = (uint8_t)(rng() & 255); } while ((a & 0x7F) == 0x7F || ((a >> 3) & 15) > 9);   // |x| < ~8, no NaN
        do { b = (uint8_t)(rng() & 255); } while ((b & 0x7F) == 0x7F || ((b >> 3) & 15) > 9);
        k8[i] = a; v8[i] = b;
        k16[i] = f2bf(nd(rng)); v16[i] = f2bf(nd(rng));
    }
    // slots: valid-first rows with random holes, some empty rows, some rows with valid entries after -1 holes
    std::vector<int32_t> sl((size_t)R * T);
    std::uniform_int_distribution<int> slot_d(0, N - 1);
    for (int r = 0; r < R; ++r) {
        int nvalid = (r % 7 == 3) ? 0 : (r % 5 == 1 ? T / 3 : T - (int)(rng() % 4));
        for (int j = 0; j < T; ++j) {
            int s = j < nvalid ? slot_d(rng) : -1;
            if (r % 4 == 2 && (j % 11) == 5) s = -1;                       // holes
            if (r % 6 == 4 && j == T - 1) s = slot_d(rng);                  // valid after the -1 tail
            sl[(size_t)r * T + j] = s;
        }
    }
    const float scale = 1.0f / 16.0f;
    // reference (double)
    std::vector<float> ref((size_t)R * Hq * HD, 0.0f);
    std::vector<double> sc(T), acc(HD);
    for (int r = 0; r < R; ++r)
        for (int h = 0; h < Hq; ++h) {
            const int kvh = h / G;
            double mx = -1e300; bool any = false;
            for (int j = 0; j < T; ++j) {
                const int s = sl[(size_t)r * T + j];
                if (s < 0) { sc[j] = 0; continue; }
                double d = 0;
                for (int x = 0; x < HD; ++x) {
                    const size_t ki = ((size_t)s * Hk + kvh) * HD + x;
                    const float kv = c.bf16kv ? bf2f(k16[ki]) : host_e4m3(k8[ki]);
                    d += (double)bf2f(qh[((size_t)r * Hq + h) * HD + x]) * kv;
                }
                sc[j] = d * scale; mx = std::max(mx, sc[j]); any = true;
            }
            if (!any) continue;
            double L = 0; std::fill(acc.begin(), acc.end(), 0.0);
            for (int j = 0; j < T; ++j) {
                const int s = sl[(size_t)r * T + j];
                if (s < 0) continue;
                const double p = std::exp(sc[j] - mx); L += p;
                for (int x = 0; x < HD; ++x) {
                    const size_t vi = ((size_t)s * Hk + kvh) * HD + x;
                    acc[x] += p * (c.bf16kv ? bf2f(v16[vi]) : host_e4m3(v8[vi]));
                }
            }
            for (int x = 0; x < HD; ++x) ref[((size_t)r * Hq + h) * HD + x] = (float)(acc[x] / L);
        }
    // device buffers (shared USM)
    auto* dq = sycl::malloc_shared<uint16_t>(qh.size(), q);
    std::memcpy(dq, qh.data(), qh.size() * 2);
    void *dk, *dv;
    if (c.bf16kv) {
        dk = sycl::malloc_shared<uint16_t>(kvn, q); dv = sycl::malloc_shared<uint16_t>(kvn, q);
        std::memcpy(dk, k16.data(), kvn * 2); std::memcpy(dv, v16.data(), kvn * 2);
    } else {
        dk = sycl::malloc_shared<uint8_t>(kvn, q); dv = sycl::malloc_shared<uint8_t>(kvn, q);
        std::memcpy(dk, k8.data(), kvn); std::memcpy(dv, v8.data(), kvn);
    }
    auto* ds = sycl::malloc_shared<int32_t>(sl.size(), q);
    std::memcpy(ds, sl.data(), sl.size() * 4);
    auto* dout = sycl::malloc_shared<uint16_t>((size_t)R * Hq * HD, q);
    std::memset(dout, 0xFF, (size_t)R * Hq * HD * 2);
    const int kvm = c.bf16kv ? KV_BF16 : (fp8_mode ? KV_FP8H : KV_FP8);
    const int CH = kernel == 1 ? 64 : THREADS;
    const int n_chunks = (T + CH - 1) / CH;
    const int cc = (cpw <= 0 || cpw > n_chunks) ? n_chunks : cpw;
    const int n_split = (n_chunks + cc - 1) / cc;
    Args a{};
    a.q = dq; a.qs0 = (int64_t)Hq * HD; a.qs1 = HD;
    a.k = dk; a.ks0 = (int64_t)Hk * HD; a.ks1 = HD;
    a.v = dv; a.vs0 = (int64_t)Hk * HD; a.vs1 = HD;
    a.slots = ds; a.ss0 = T; a.T = T;
    const float vun = kvm == KV_FP8H ? 256.0f : 1.0f;
    a.qscale = scale * LOG2E * vun; a.vunscale = vun;
    a.cpw = cc; a.n_split = n_split; a.n_chunks = n_chunks; a.out = dout;
    float* scratch = nullptr;
    const int RB = 5;   // small row batches: exercises row0 offsets
    if (n_split > 1) {
        scratch = sycl::malloc_shared<float>((size_t)RB * Hk * n_split * G * (HD + 2), q);
        a.pacc = scratch; a.pm = scratch + (size_t)RB * Hk * n_split * G * HD; a.pl = a.pm + (size_t)RB * Hk * n_split * G;
    }
    auto launch = [&](int rows) {
        switch (kvm) {
            case KV_BF16: dispatch_sg<KV_BF16>(kernel, sg, false, q, a, rows, Hk); break;
            case KV_FP8: dispatch_sg<KV_FP8>(kernel, sg, false, q, a, rows, Hk); break;
            default: dispatch_sg<KV_FP8H>(kernel, sg, false, q, a, rows, Hk); break;
        }
    };
    if (n_split == 1) { a.row0 = 0; launch(R); }
    else {
        for (int r0 = 0; r0 < R; r0 += RB) {
            const int rows = std::min(RB, R - r0);
            a.row0 = r0;
            launch(rows);
            MergeKernel mk{a.pacc, a.pm, a.pl, a.out, r0, n_split, Hk, a.vunscale};
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t)rows * Hq, HD), sycl::range<2>(1, HD)), mk);
            });
            q.wait();
            if (std::getenv("N108_DBG") && r0 == 0 && 0) {
                for (int rr = 0; rr < rows; ++rr) for (int sp_ = 0; sp_ < n_split; ++sp_)
                    std::printf("dbg row %d split %d: m=%g l=%g acc0=%g\n", rr, sp_, a.pm[(rr * Hk * n_split + sp_) * G], a.pl[(rr * Hk * n_split + sp_) * G], a.pacc[(size_t)(rr * Hk * n_split + sp_) * G * HD]);
                // host: per-split max (log2 units) and sums for row 0 head 0
                for (int sp_ = 0; sp_ < n_split; ++sp_) {
                    double mx = -1e300, L = 0, A = 0;
                    std::vector<double> scv;
                    for (int j = sp_ * cc * CH; j < std::min(T, (sp_ + 1) * cc * CH); ++j) {
                        int s = sl[j]; if (s < 0) continue;
                        double d = 0;
                        for (int x = 0; x < HD; ++x) d += (double)bf2f(qh[x]) * (c.bf16kv ? bf2f(k16[(size_t)s * Hk * HD + x]) : host_e4m3(k8[(size_t)s * Hk * HD + x]));
                        d *= scale * 1.4426950408889634; scv.push_back(d); mx = std::max(mx, d);
                    }
                    std::printf("host split %d: m=%g n=%zu\n", sp_, mx, scv.size());
                }
            }
        }
    }
    q.wait();
    double maxd = 0, sumd = 0; int bad_zero = 0, nonfinite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float o = bf2f(dout[i]);
        if (!std::isfinite(o)) { ++nonfinite; continue; }
        const double d = std::fabs((double)o - ref[i]);
        maxd = std::max(maxd, d); sumd += d;
    }
    if (std::getenv("N108_DBG"))
        for (int r = 0; r < R; ++r) {
            double md = 0;
            for (int i = 0; i < Hq * HD; ++i) md = std::max(md, std::fabs((double)bf2f(dout[(size_t)r * Hq * HD + i]) - ref[(size_t)r * Hq * HD + i]));
            int nv = 0; for (int j = 0; j < T; ++j) nv += sl[(size_t)r * T + j] >= 0;
            std::printf("  row %d nvalid %d max_abs %.3e\n", r, nv, md);
        }
    for (int r = 0; r < R; ++r) {
        bool empty = true;
        for (int j = 0; j < T; ++j) empty &= sl[(size_t)r * T + j] < 0;
        if (empty) for (int i = 0; i < Hq * HD; ++i) bad_zero += dout[(size_t)r * Hq * HD + i] != 0;
    }
    const bool ok = maxd <= 2e-2 && sumd / ref.size() <= 2e-3 && bad_zero == 0 && nonfinite == 0;
    std::printf("kernel=%d cpw=%d(split %d) sg=%d kv=%s fp8=%d R=%d T=%d: max_abs=%.3e mean_abs=%.3e bad_zero=%d nonfinite=%d %s\n",
                kernel, cpw, n_split, sg, c.bf16kv ? "bf16" : "fp8", fp8_mode, R, T, maxd, sumd / ref.size(), bad_zero,
                nonfinite, ok ? "OK" : "FAIL");
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(ds, q); sycl::free(dout, q);
    if (scratch) sycl::free(scratch, q);
    return ok ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};   // like torch XPU streams
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    {   // decode routes vs host
        auto* c = sycl::malloc_shared<uint8_t>(256, q);
        auto* o = sycl::malloc_shared<float>(512, q);
        for (int i = 0; i < 256; ++i) c[i] = (uint8_t)i;
        q.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
            o[i] = load1<KV_FP8>(c + i);
            o[256 + i] = load1<KV_FP8H>(c + i) * 256.0f;
        }).wait();
        int bad = 0;
        for (int i = 0; i < 256; ++i) {
            if ((i & 0x7F) == 0x7F) continue;
            bad += o[i] != host_e4m3((uint8_t)i) || std::signbit(o[i]) != std::signbit(host_e4m3((uint8_t)i));
            bad += o[256 + i] != host_e4m3((uint8_t)i);
        }
        std::printf("fp8 decode: %d mismatches %s\n", bad, bad ? "FAIL" : "OK");
        sycl::free(c, q); sycl::free(o, q);
        if (bad) return 1;
    }
    int fails = 0;
    if (std::getenv("N108_DBG")) {
        Case d2{7, 1, 128, 512, true};
        return run_case(q, d2, 1, 1, 32, 0, 5); }
    const Case small{13, 2, 300, 4096, false}, full{9, 2, 2051, 8192, false}, bf{9, 2, 700, 4096, true};
    for (int sg : {16, 32})
        for (int kernel : {1, 2})
            for (int cpw : {0, 1, 3}) {
                fails += run_case(q, small, kernel, cpw, sg, 1, 11);
                if (cpw != 3) fails += run_case(q, small, kernel, cpw, sg, 0, 12);
            }
    for (int kernel : {1, 2}) {
        fails += run_case(q, full, kernel, 0, 16, 1, 21);
        fails += run_case(q, full, kernel, 1, 32, 1, 22);
        fails += run_case(q, bf, kernel, 0, 16, 0, 31);
        fails += run_case(q, bf, kernel, 2, 32, 0, 32);
    }
    std::printf("CPU TEST %s (%d failing cases)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

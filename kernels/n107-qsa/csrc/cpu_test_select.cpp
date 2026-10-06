// N108 CPU check of the selection kernels (OpenCL CPU device in image 24c872759256, no torch):
//  * device scores vs a double-precision host reference (relative error),
//  * device top-k == host exact top-k of the DEVICE scores with the same rule (value desc, ties -> lowest index),
//    bit-for-bit, including rows with len <= topk, heavy ties (quantised scores) and offsets (row_starts > 0).
//   source /opt/intel/oneapi/setvars.sh; icpx -fsycl -O2 -std=c++17 csrc/cpu_test_select.cpp -o build/cpu_test_select
#define N108_NO_TORCH
#include "qsa_select_sycl.sycl"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace n108sel;

static uint16_t f2bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f; }

static int run(sycl::queue& q, int R, int C, int topk, bool ties, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> qh((size_t)R * IH * ID), kh((size_t)C * ID);
    for (auto& x : qh) x = f2bf(ties ? (float)(int)(nd(rng) * 2) : nd(rng));
    for (auto& x : kh) x = f2bf(ties ? (float)(int)(nd(rng) * 2) : nd(rng));
    std::vector<int32_t> st(R), en(R);
    const int base = 7;   // second "sequence": keys start at an offset
    for (int r = 0; r < R; ++r) {
        st[r] = r % 3 == 0 ? 0 : base;
        const int vis = (r * (C - base)) / R + (r % 5);
        en[r] = std::min(C, st[r] + vis);
    }
    auto* dq = sycl::malloc_shared<uint16_t>(qh.size(), q); std::memcpy(dq, qh.data(), qh.size() * 2);
    auto* dk = sycl::malloc_shared<uint16_t>(kh.size(), q); std::memcpy(dk, kh.data(), kh.size() * 2);
    auto* ds = sycl::malloc_shared<int32_t>(R, q); std::memcpy(ds, st.data(), R * 4);
    auto* de = sycl::malloc_shared<int32_t>(R, q); std::memcpy(de, en.data(), R * 4);
    auto* sc = sycl::malloc_shared<float>((size_t)R * C, q);
    auto* out = sycl::malloc_shared<int32_t>((size_t)R * topk, q);
    for (size_t i = 0; i < (size_t)R * C; ++i) sc[i] = NAN;
    const float inv = (float)(1.0 / std::sqrt(128.0));
    launch_scores<8>(q, dq, IH * ID, ID, dk, ID, ds, de, sc, C, R, C, inv);
    launch_topk(q, sc, C, ds, de, out, topk, R);
    q.wait();
    double max_rel = 0; int bad_rows = 0, unwritten = 0;
    for (int r = 0; r < R; ++r) {
        const int len = std::max(0, en[r] - st[r]);
        for (int j = 0; j < len; ++j) {
            const int n = st[r] + j;
            double tot = 0, mag = 0;
            for (int h = 0; h < IH; ++h) {
                double d = 0, a = 0;
                for (int x = 0; x < ID; ++x) {
                    const double p = (double)bf2f(qh[((size_t)r * IH + h) * ID + x]) * bf2f(kh[(size_t)n * ID + x]);
                    d += p; a += std::fabs(p);
                }
                tot += std::max(d, 0.0); mag += a;
            }
            tot *= 1.0 / std::sqrt(128.0); mag *= 1.0 / std::sqrt(128.0);
            const float dv = sc[(size_t)r * C + j];
            if (std::isnan(dv)) { ++unwritten; continue; }
            max_rel = std::max(max_rel, std::fabs(dv - tot) / (mag + 1e-30));
        }
        // host exact top-k of the device scores
        std::vector<int> idx(len);
        std::iota(idx.begin(), idx.end(), 0);
        auto key = [&](int j) { return sc[(size_t)r * C + j] + 0.0f; };
        std::vector<int> want;
        if (len <= topk) want = idx;
        else {
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return key(a) > key(b); });
            want.assign(idx.begin(), idx.begin() + topk);
            std::sort(want.begin(), want.end());
        }
        std::vector<int> got;
        for (int j = 0; j < topk; ++j) { const int v = out[(size_t)r * topk + j]; if (v >= 0) got.push_back(v); }
        const bool asc = std::is_sorted(got.begin(), got.end());
        bool tail_ok = true;
        for (int j = (int)got.size(); j < topk; ++j) tail_ok &= out[(size_t)r * topk + j] == -1;
        if (got != want || !asc || !tail_ok) ++bad_rows;
    }
    std::printf("R=%d C=%d topk=%d ties=%d: score max_rel_err=%.3e unwritten=%d topk_bad_rows=%d %s\n", R, C, topk, ties,
                max_rel, unwritten, bad_rows, (max_rel < 3e-5 && !unwritten && !bad_rows) ? "OK" : "FAIL");
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(ds, q); sycl::free(de, q); sycl::free(sc, q); sycl::free(out, q);
    return (max_rel < 3e-5 && !unwritten && !bad_rows) ? 0 : 1;
}

int main() {
    sycl::queue q{sycl::cpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int f = 0;
    f += run(q, 37, 700, 64, false, 1);
    f += run(q, 37, 700, 64, true, 2);
    f += run(q, 20, 2048, 512, false, 3);
    f += run(q, 20, 2048, 512, true, 4);
    f += run(q, 9, 300, 512, false, 5);
    std::printf("CPU SELECT TEST %s (%d failing)\n", f ? "FAIL" : "PASS", f);
    return f ? 1 : 0;
}

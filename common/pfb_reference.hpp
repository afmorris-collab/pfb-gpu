// fp64 reference model of the PFB in pfb.cuh, for tests and benchmarks (host code only).
//
// Output row r, ADC channel a, is the FFT of the M-sample frame
//     y[c]       = sum_{k even} taps[k][c] * x[r+k][c]     c = 0..M/2-1, k = 0..2T-1
//     y[c + M/2] = sum_{k odd}  taps[k][c] * x[r+k][c]
// (x = offset-binary sample - 32768), with the two halves swapped on odd r, and
//     X[bin] = sum_n y[n] e^{-2 pi i bin n / M},   bin = 0..M/2.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <vector>

struct PfbCheckResult {
    size_t count = 0;            // values compared
    size_t nonfinite = 0;        // NaN/Inf in the checked output (e.g. never written)
    double rms = 0;              // RMS of the model output
    double rel_rms_err = 0;      // RMS error / rms
    double max_abs_err = 0;
    bool pass = false;           // all finite, rel RMS err < 1e-5, max err < 1e-3 * rms
};

// Compares out(a, bin, row) -> std::complex<float> with the model, for ADC channels 0..3, the
// given bins (all if empty) and a sample of rows: the first and last 64 and 128 spread over Q.
//   in:   (Q - 1 + 2T) * M/2 samples, each 4 interleaved uint16 (one per ADC channel)
//   taps: 2T rows of M/2
template <typename Out>
PfbCheckResult PfbModelCheck(int M, int T, int Q, const uint16_t* in, const float* taps, Out out,
                             std::vector<int> bins = {})
{
    if (bins.empty())
        for (int b = 0; b <= M / 2; b++) bins.push_back(b);
    std::vector<std::complex<double>> tw(M);
    for (int n = 0; n < M; n++) tw[n] = std::polar(1.0, -2.0 * M_PI * n / M);

    PfbCheckResult res;
    double sum_ref = 0, sum_err = 0;
    std::vector<double> y(M);
    const int stride = std::max(1, Q / 128);
    for (int r = 0; r < Q; r++) {
        if (!(r < 64 || r >= Q - 64 || r % stride == 0)) continue;
        for (int a = 0; a < 4; a++) {
            for (int c = 0; c < M / 2; c++) {
                double h[2] = {};
                for (int k = 0; k < 2 * T; k++)
                    h[k & 1] += (double)taps[k * (M / 2) + c] *
                                ((int)in[((size_t)(r + k) * (M / 2) + c) * 4 + a] - 32768);
                y[c] = h[r & 1];
                y[c + M / 2] = h[(r & 1) ^ 1];
            }
            for (int b : bins) {
                std::complex<double> X = 0;
                for (int n = 0; n < M; n++) X += y[n] * tw[(size_t)b * n % M];
                const std::complex<float> g = out(a, b, r);
                res.count++;
                sum_ref += std::norm(X);
                if (!std::isfinite(g.real()) || !std::isfinite(g.imag())) { res.nonfinite++; continue; }
                const double e = std::abs(std::complex<double>(g.real(), g.imag()) - X);
                sum_err += e * e;
                res.max_abs_err = std::max(res.max_abs_err, e);
            }
        }
    }
    res.rms = std::sqrt(sum_ref / std::max<size_t>(res.count, 1));
    res.rel_rms_err = std::sqrt(sum_err / std::max<size_t>(res.count, 1)) / res.rms;
    res.pass = res.count > 0 && res.nonfinite == 0 && res.rel_rms_err < 1e-5 && res.max_abs_err < 1e-3 * res.rms;
    return res;
}

inline void PfbPrintCheck(const char* what, const PfbCheckResult& res)
{
    printf("  %-40s %s (rel RMS err %.2e, max abs err %.3g of RMS %.3g, %zu values, %zu nonfinite)\n",
           what, res.pass ? "PASS" : "FAIL", res.rel_rms_err, res.max_abs_err, res.rms, res.count, res.nonfinite);
}

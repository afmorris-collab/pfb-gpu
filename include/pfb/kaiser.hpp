// Kaiser-windowed low-pass prototype filter for the PFB.
#pragma once

#include <cmath>
#include <vector>

namespace pfb {
namespace detail {

// Zeroth-order modified Bessel function of the first kind, I0(x), by its power series.
inline double BesselI0(double x) {
    double sum = 1.0, u = 1.0;
    for (int i = 1; i < 50; ++i) {
        u *= x / 2.0 / i;
        sum += u * u;
        if (u * u < 1e-15 * sum) break;
    }
    return sum;
}

// Kaiser window beta for a stopband attenuation in dB (Kaiser's empirical formula).
inline double KaiserBeta(double atten_db) {
    if (atten_db > 50.0) return 0.1102 * (atten_db - 8.7);
    if (atten_db >= 21.0) return 0.5842 * std::pow(atten_db - 21.0, 0.4) + 0.07886 * (atten_db - 21.0);
    return 0.0;
}

}  // namespace detail

// Default stopband attenuation of the PFB prototype filter, in dB.
constexpr double DEFAULT_ATTEN_DB = 120;

// Low-pass FIR taps by the Kaiser window method, normalized to unit DC gain.
//   ntaps:    number of taps (T*M for the PFB)
//   atten_db: stopband attenuation in dB (normally DEFAULT_ATTEN_DB)
//   cutoff:   cutoff frequency in cycles per sample, 0..0.5 (1/M for the PFB)
inline std::vector<float> KaiserTaps(int ntaps, double atten_db, double cutoff) {
    const double beta = detail::KaiserBeta(atten_db);
    const double i0_beta = detail::BesselI0(beta);
    const double center = (ntaps - 1) / 2.0;
    const double wc = 2.0 * M_PI * cutoff;                  // cutoff in radians per sample

    std::vector<float> taps(ntaps);
    double sum = 0.0;
    for (int i = 0; i < ntaps; ++i) {
        const double n = i - center;
        const double sinc = std::abs(n) < 1e-9 ? wc / M_PI : std::sin(wc * n) / (M_PI * n);
        const double r = n / center;
        const double window = detail::BesselI0(beta * std::sqrt(1.0 - r * r)) / i0_beta;
        taps[i] = sinc * window;
        sum += taps[i];
    }
    for (float& h : taps) h /= sum;
    return taps;
}

}  // namespace pfb

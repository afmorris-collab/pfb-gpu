// PFB correctness test for this build's M and T (CMake PFB_M, PFB_T), run through pfb::Channelizer.
// Exit code 0 means every check passed.
//   1. All bins, packed [adc][bin][row] output: Channelizer output vs the fp64 model
//      (pfb_reference.hpp).
//   2. Every third bin selected with set_bins, into a strided output laid out like the HAARP pfb
//      service's ring buffers ([bin][source][row], source = 4*card + adc, padded rows; here one
//      card): those bins match the model and every other element of the host buffer is left
//      untouched.
//   3. Bad arguments (M, T, taps, Q, bins, output strides) throw std::invalid_argument, including
//      a Q too large for the separate path's 32-bit indexing.
//   4. Output vs tests/data/golden_M<M>_T<T>.bin, recorded with the HAARP pfb service's taps
//      (pfb::KaiserTaps at DEFAULT_ATTEN_DB, scaled for int16 input); skipped if there is no file
//      for this build's M and T. "test_pfb --generate" writes the file.

#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "pfb/channelizer.hpp"
#include "pfb/kaiser.hpp"
#include "pfb_reference.hpp"
#include "bench.hpp"

using namespace std;
using cplx = complex<float>;

constexpr int M = PFB_M, T = PFB_T;
constexpr int NBINS = M / 2 + 1;
using pfb::NADC;

// Host output for one digitizer card with the given strides, initialized to NaN.
struct HostOutput {
    vector<cplx> data;
    pfb::PfbOutput out;
    HostOutput(size_t bin_stride, size_t adc_stride, size_t size)
        : data(size, cplx(NAN, NAN)), out{data.data(), bin_stride, adc_stride} {}
    cplx& at(int a, int b, int r) { return data[a * out.adc_stride + b * out.bin_stride + r]; }
    static HostOutput Packed(int Q) { return HostOutput(Q, (size_t)NBINS * Q, (size_t)NADC * NBINS * Q); }
};

// Checks 1 and 2: random input and taps, Q = 2000 rows.
static bool ModelTests() {
    const int Q = 2000;                       // multiple of 8 (fused) and 10 (separate)
    const vector<float> taps = RandomTaps(M, T);
    pfb::Channelizer c(M, T, Q, taps);
    vector<uint16_t> input = RandomInput(4 * c.input_samples());

    HostOutput all = HostOutput::Packed(Q);
    c.process(input.data(), all.out);
    const PfbCheckResult r1 = PfbModelCheck(M, T, Q, input.data(), taps.data(),
                                            [&](int a, int b, int r) { return all.at(a, b, r); });
    PfbPrintCheck("all bins vs fp64 model", r1);

    vector<int> bins;
    for (int b = 0; b < NBINS; b += 3) bins.push_back(b);
    c.set_bins(bins);
    const size_t pitch = Q + 64;              // ring-buffer-like: bins outermost, padded rows
    HostOutput sel(NADC * pitch, pitch, (size_t)NBINS * NADC * pitch);
    c.process(input.data(), sel.out);
    const PfbCheckResult r2 = PfbModelCheck(M, T, Q, input.data(), taps.data(),
                                            [&](int a, int b, int r) { return sel.at(a, b, r); }, bins);
    PfbPrintCheck("every third bin, strided vs fp64 model", r2);
    size_t written = 0;                       // everything but the selected rows must still be NaN
    vector<bool> expected(sel.data.size(), false);
    for (int b : bins)
        for (int a = 0; a < NADC; a++)
            for (int r = 0; r < Q; r++) expected[a * sel.out.adc_stride + b * sel.out.bin_stride + r] = true;
    for (size_t i = 0; i < sel.data.size(); i++) written += !expected[i] && !std::isnan(sel.data[i].real());
    printf("  %-40s %s (%zu values written)\n", "other elements left untouched", written ? "FAIL" : "PASS", written);
    return r1.pass && r2.pass && written == 0;
}

// Check 3: argument validation.
static bool ArgumentTests() {
    const int Q = 40;
    const vector<float> taps((size_t)T * M, 0.f);
    auto throws = [](const function<void()>& f) {
        try { f(); } catch (const invalid_argument&) { return true; }
        return false;
    };
    pfb::Channelizer c(M, T, Q, taps);
    HostOutput out = HostOutput::Packed(Q);
    vector<uint16_t> input(4 * c.input_samples(), 32768);
    const bool pass =
        throws([&] { pfb::Channelizer(M + 2, T, Q, vector<float>((size_t)T * (M + 2))); }) &&
        throws([&] { pfb::Channelizer(M, T + 2, Q, vector<float>((size_t)(T + 2) * M)); }) &&
        throws([&] { pfb::Channelizer(M, T, Q, vector<float>((size_t)T * M - 1)); }) &&
        throws([&] { pfb::Channelizer(M, T, Q + 1, taps); }) &&
        throws([&] { pfb::Channelizer(M, T, 0, taps); }) &&
        (pfb::UsesFused(M) || throws([&] { pfb::Channelizer(M, T, (INT_MAX / (4 * M) / 10 + 1) * 10, taps); })) &&
        throws([&] { c.set_bins({0, NBINS}); }) &&
        throws([&] { c.set_bins({-1}); }) &&
        throws([&] { c.process(input.data(), {nullptr, (size_t)Q, (size_t)NBINS * Q}); }) &&
        throws([&] { c.process(input.data(), {out.data.data(), (size_t)Q - 1, (size_t)NBINS * Q}); }) &&
        throws([&] { c.process(input.data(), {out.data.data(), (size_t)Q, (size_t)Q - 1}); });
    printf("  %-40s %s\n", "bad arguments throw invalid_argument", pass ? "PASS" : "FAIL");
    return pass;
}

// Check 4: golden output for this build's M and T.
static bool GoldenTest(bool generate) {
    const int Q = pfb::UsesFused(M) ? 40 : 50;    // a multiple of 8 (fused) or 10 (separate)
    const string golden_file = string(PFB_TEST_DATA_DIR) + "/golden_M" + to_string(M) + "_T" + to_string(T) + ".bin";
    ifstream ifs(golden_file, ios::binary);
    if (!generate && !ifs) {
        printf("  %-40s SKIPPED (no %s)\n", "golden output", golden_file.c_str());
        return true;
    }
    // Unit-gain Kaiser filter at DEFAULT_ATTEN_DB, scaled by 2/16384: the scale the golden files
    // were recorded with. (The HAARP pfb service now uses 1/16384; the scale doesn't matter here.)
    vector<float> taps = pfb::KaiserTaps(T * M, pfb::DEFAULT_ATTEN_DB, 1. / M);
    for (float& h : taps) h *= 2.f / 16384;
    pfb::Channelizer c(M, T, Q, taps);
    vector<uint16_t> input(c.input_samples() * 4);
    for (size_t i = 0; i < input.size(); i++) input[i] = (i * 137) % 65536;
    HostOutput out = HostOutput::Packed(Q);
    c.process(input.data(), out.out);

    // File layout: for each bin, Q samples of ADC channel 0, then 1, 2, 3. Only output that
    // matches the fp64 model is recorded.
    if (generate) {
        const PfbCheckResult res = PfbModelCheck(M, T, Q, input.data(), taps.data(),
                                                 [&](int a, int b, int r) { return out.at(a, b, r); });
        PfbPrintCheck("golden output vs fp64 model", res);
        if (!res.pass) return false;
        ofstream ofs(golden_file, ios::binary);
        for (int b = 0; b < NBINS; b++)
            for (int a = 0; a < NADC; a++) ofs.write(reinterpret_cast<char*>(&out.at(a, b, 0)), Q * sizeof(cplx));
        printf("  wrote %s\n", golden_file.c_str());
        return bool(ofs);
    }
    vector<cplx> golden(Q);
    float max_err = 0;
    for (int b = 0; b < NBINS; b++)
        for (int a = 0; a < NADC; a++) {
            ifs.read(reinterpret_cast<char*>(golden.data()), Q * sizeof(cplx));
            for (int r = 0; r < Q; r++) max_err = max(max_err, abs(out.at(a, b, r) - golden[r]));
        }
    const bool pass = bool(ifs) && max_err < 1e-4;
    printf("  %-40s %s (max abs err %.3g)\n", "golden output", pass ? "PASS" : "FAIL", max_err);
    return pass;
}

static int Main(int argc, char** argv) {
    const bool generate = argc > 1 && string(argv[1]) == "--generate";
    printf("test_pfb: M=%d T=%d (%s path)\n", M, T, pfb::UsesFused(M) ? "fused" : "separate");
    bool pass = true;
    if (!generate) pass &= ModelTests() & ArgumentTests();
    pass &= GoldenTest(generate);
    printf("%s\n", pass ? "TEST PASSED" : "TEST FAILED");
    return pass ? 0 : 1;
}

// Library errors (bad arguments, CUDA or cuFFT failures) arrive as exceptions.
int main(int argc, char** argv)
{
    try {
        return Main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "test_pfb: %s\n", e.what());
        return 1;
    }
}

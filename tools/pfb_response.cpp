// Frequency response of the PFB at this build's M and T (CMake PFB_M, PFB_T), measured through
// pfb::Channelizer with synthetic int16 tones and written to CSV: the numerical counterpart of
// sweeping a signal generator into the ADCs (haarp's siggen_tone_power.py).
//
// Bin k sees a real input tone at f as two components, |H(f - k)| and its mirror at -f (f in
// channel spacings fs/M; bin k is centered at k). Each bin's output is at baseband, sampled at
// 2 fs/M, so the tone's own component lands at output frequency (f - k) wrapped to -1..1. The
// response is its complex amplitude at that frequency, averaged over Q rows (rectangular window,
// exact on the grid below) and divided by the tone's A/2. Tones are placed on a grid of 1/Q
// channels but never at a bin center, so the mirror lands on a zero of the average and the
// measurement is H alone (also in the DC and Nyquist bins). With Q = 1000, averaging brings the
// int16 rounding noise floor down to about -135 dB (M=64) / -146 dB (M=800).
//
// Two modes:
//
//   pfb_response [output directory]       (default: current directory; created if missing)
//     Writes, in channel spacings:
//       response_M<M>_T<T>_detail.csv   ~500 tones across one channel around bin M/8; bins within
//                                       4 of it, so offsets -4.5..4.5 at 2/Q resolution
//       response_M<M>_T<T>_band.csv     a tone every 1/8 channel from DC to Nyquist; bins within 3
//     Columns: tone, bin, offset (= tone - bin), out_freq (where the tone lands in the bin's
//     output), measured_db, predicted_db (|H(offset)| from the taps in fp64).
//
//   pfb_response --bin K --start HZ --stop HZ --step HZ [options]
//     Sweeps a tone from --start to --stop (inclusive) and measures it in bin K only, in Hz like
//     siggen_tone_power.py. Prints a table; --csv also writes it to a file.
//       --fs HZ       ADC sample rate (default 80e6, as at HAARP); bin K is centered at K fs/M
//       --amp A       tone amplitude in ADC counts (default 32000; int16 full scale is 32767)
//       --rows Q      output rows averaged per tone (default 1000; a multiple of QMultiple(M)).
//                     Tones are rounded to the grid fs/(M Q); one on a bin center (or DC, or
//                     Nyquist) is moved one grid step, and the frequency actually used is reported.
//       --scale S     tap gain (default 1, unit DC gain; haarp's pfb service uses 1/16384). Only
//                     a_rms and power_db depend on it.
//       --csv FILE    also write the results to FILE
//     Columns: freq_hz, offset_from_center_hz (f - K fs/M), aliased_offset_hz (where the tone
//     lands in the bin's output, -fs/M..fs/M), a_rms (tone amplitude in the bin's output, as
//     siggen_tone_power.py reports it), power_db (20 log10 a_rms), gain_db (a_rms relative to
//     the input tone's A/2 and the tap gain, i.e. |H|), predicted_db (|H| from the taps in fp64).
//
// All modes use the default taps: pfb::KaiserTaps at DEFAULT_ATTEN_DB, unit DC gain.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#include "pfb/channelizer.hpp"
#include "pfb/kaiser.hpp"

#include <functional>

using namespace std;

constexpr int M = PFB_M, T = PFB_T;
constexpr int NBINS = M / 2 + 1;
using pfb::NADC;

// One measurement: a tone (on the 1/Q-channel grid) seen in one bin.
struct Point {
    long grid;          // tone frequency in 1/Q channel spacings
    long bin;
    long offset;        // grid - bin * Q
    double out_freq;    // where the tone lands in the bin's output, in channel spacings, -1..1
    double a_rms;       // tone amplitude in the bin's output (output units)
    double gain;        // a_rms / (A/2) / tap gain: measured |H(offset)|
    double predicted;   // |H(offset)| from the taps in fp64
};

// Wrap a frequency in channel spacings into a bin's output band, -1..1.
static double Wrap(double f) { return f - 2 * floor((f + 1) / 2); }

static double Db(double amplitude) { return 20 * log10(max(amplitude, 1e-20)); }

static vector<float> Scaled(vector<float> taps, double scale)
{
    for (float& h : taps) h = (float)(h * scale);
    return taps;
}

class Sweeper {
public:
    // taps: unit DC gain; the Channelizer runs with taps * scale.
    Sweeper(const vector<float>& taps, int Q, double amp, double scale)
        : Q(Q), amp(amp), scale(scale), taps(taps.begin(), taps.end()),
          c(M, T, Q, Scaled(taps, scale)), input(NADC * c.input_samples()),
          output((size_t)NADC * NBINS * Q), view{output.data(), (size_t)Q, (size_t)NBINS * Q} {}

    // Measure tones (up to 4 at once, one per ADC channel). For each tone, measure bin `bin` if
    // it is >= 0, else every bin within `span` of the tone, and pass each result to `sink`.
    void Run(const vector<long>& tones, int span, int bin, const function<void(const Point&)>& sink)
    {
        for (size_t i = 0; i < tones.size(); i += NADC) {
            const size_t n = min((size_t)NADC, tones.size() - i);
            for (size_t a = 0; a < NADC; a++) Generate(a, a < n ? tones[i + a] : tones[i]);
            c.process(input.data(), view);
            for (size_t a = 0; a < n; a++) {
                const long k0 = lround((double)tones[i + a] / Q);
                const long lo = bin >= 0 ? bin : max(0L, k0 - span);
                const long hi = bin >= 0 ? bin : min((long)NBINS - 1, k0 + span);
                for (long k = lo; k <= hi; k++) sink(Measure(a, tones[i + a], k));
            }
        }
    }

private:
    // Tone on ADC channel a: offset binary, with triangular dither of +-1 LSB.
    void Generate(size_t a, long grid)
    {
        const double f = (double)grid / Q / M;                      // cycles per sample
        const complex<double> step = polar(1.0, 2 * M_PI * f);
        complex<double> z = polar(1.0, 0.1 * a);                    // phase differs per channel
        const size_t nsamp = c.input_samples();
        for (size_t s = 0; s < nsamp; s++) {
            const double dither = Uniform() - Uniform();
            const double v = lround(32768 + amp * z.real() + dither);
            input[s * NADC + a] = (uint16_t)min(65535.0, max(0.0, v));
            z *= step;
            if ((s & 1023) == 0) z /= abs(z);                       // keep |z| = 1
        }
    }

    Point Measure(size_t a, long grid, long k)
    {
        const long off = grid - k * Q;                              // offset in 1/Q channels
        const double out_freq = Wrap((double)off / Q);
        // Average of y[r] exp(-2 pi i nu r): the tone advances out_freq / 2 cycles per row.
        const complex<double> step = polar(1.0, -M_PI * out_freq);
        complex<double> z = 1, sum = 0;
        const complex<float>* y = output.data() + a * view.adc_stride + k * view.bin_stride;
        for (int r = 0; r < Q; r++) {
            sum += complex<double>(y[r]) * z;
            z *= step;
        }
        const double a_rms = abs(sum) / Q;
        return {grid, k, off, out_freq, a_rms, a_rms / (amp / 2) / scale, Predicted((double)off / Q)};
    }

    // |H(offset)| (offset in channel spacings) from the unit-gain taps in fp64. The taps are
    // symmetric, so H is real up to linear phase: sum of h[n] cos(2 pi x (n - center)), x in
    // cycles/sample. The cosines come from a rotating phasor, renormalized now and then.
    double Predicted(double offset) const
    {
        const double x = offset / M, center = (taps.size() - 1) / 2.0;
        const complex<double> step = polar(1.0, 2 * M_PI * x);
        complex<double> z = polar(1.0, -2 * M_PI * x * center);
        double h = 0;
        for (size_t n = 0; n < taps.size(); n++) {
            h += taps[n] * z.real();
            z *= step;
            if ((n & 1023) == 1023) z /= abs(z);
        }
        return fabs(h);
    }

    double Uniform() {                                              // xorshift64*, in [0, 1)
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        return (double)((rng * 2685821657736338717ULL) >> 11) / 9007199254740992.0;
    }

    const int Q;
    const double amp, scale;
    const vector<double> taps;
    pfb::Channelizer c;
    vector<uint16_t> input;
    vector<complex<float>> output;
    pfb::PfbOutput view;
    uint64_t rng = 88172645463325252ULL;
};

static FILE* OpenCsv(const string& path, const char* header)
{
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { perror(path.c_str()); exit(1); }
    fprintf(f, "%s\n", header);
    return f;
}

static vector<float> DefaultTaps() { return pfb::KaiserTaps(T * M, pfb::DEFAULT_ATTEN_DB, 1. / M); }

// ---- pfb_response [output directory] ----------------------------------------------------------
static int Standard(const string& dir)
{
    constexpr int Q = 1000;                 // multiple of 8 and 10, grid of 1/Q channels
    std::filesystem::create_directories(dir);
    const string prefix = dir + "/response_M" + to_string(M) + "_T" + to_string(T);
    printf("pfb_response: M=%d T=%d (%s path), Q=%d rows per tone\n", M, T, pfb::UsesFused(M) ? "fused" : "separate", Q);
    Sweeper sweeper(DefaultTaps(), Q, 32000, 1);
    const char* header = "tone,bin,offset,out_freq,measured_db,predicted_db";
    auto writer = [](FILE* f) {
        return [f](const Point& p) {
            fprintf(f, "%.6f,%ld,%.6f,%.6f,%.3f,%.3f\n", (double)p.grid / Q, p.bin, (double)p.offset / Q,
                    p.out_freq, Db(p.gain), Db(p.predicted));
        };
    };

    // Detail: tones at k0 - 0.5 + (2j+1)/Q, never at a bin center.
    const long k0 = M / 8;
    vector<long> detail;
    for (long j = 0; j < Q / 2; j++) detail.push_back(k0 * Q - Q / 2 + 2 * j + 1);
    FILE* f = OpenCsv(prefix + "_detail.csv", header);
    sweeper.Run(detail, 4, -1, writer(f));
    fclose(f);
    printf("  wrote %s_detail.csv (%zu tones around bin %ld)\n", prefix.c_str(), detail.size(), k0);

    // Band: a tone every 1/8 channel from DC to Nyquist, shifted 1/Q off the bin centers.
    vector<long> band;
    for (long g = 1; g < (long)(M / 2) * Q; g += Q / 8) band.push_back(g);
    f = OpenCsv(prefix + "_band.csv", header);
    sweeper.Run(band, 3, -1, writer(f));
    fclose(f);
    printf("  wrote %s_band.csv (%zu tones, DC to Nyquist)\n", prefix.c_str(), band.size());
    return 0;
}

// ---- pfb_response --bin K --start HZ --stop HZ --step HZ [options] ---------------------------
struct SweepArgs {
    int bin = -1;
    double start = NAN, stop = NAN, step = NAN;
    double fs = 80e6, amp = 32000, scale = 1;
    int rows = 1000;
    string csv;
};

static int Sweep(const SweepArgs& s)
{
    if (s.bin < 0 || s.bin >= NBINS)
        throw invalid_argument("--bin must be in 0.." + to_string(NBINS - 1));
    if (std::isnan(s.start) || std::isnan(s.stop) || std::isnan(s.step))
        throw invalid_argument("--bin needs --start, --stop and --step");
    if (!(s.fs > 0) || !(s.amp > 0 && s.amp <= 32767) || !(s.scale > 0))
        throw invalid_argument("--fs and --scale must be positive, --amp in (0, 32767]");
    if (!(s.start >= 0 && s.start <= s.stop && s.stop <= s.fs / 2 && s.step > 0))
        throw invalid_argument("need 0 <= --start <= --stop <= fs/2 and --step > 0");
    const int Q = s.rows;
    if (Q <= 0 || Q % pfb::QMultiple(M))
        throw invalid_argument("--rows must be a positive multiple of " + to_string(pfb::QMultiple(M)));
    const double chan = s.fs / M;           // channel spacing (Hz)
    const double res = chan / Q;            // tone grid (Hz)
    if (s.step < res)
        throw invalid_argument("--step is below the frequency resolution fs/(M rows) = " + to_string(res) +
                               " Hz; increase --rows");

    vector<long> tones;
    int moved = 0;
    for (long i = 0;; i++) {
        const double f = s.start + i * s.step;
        if (f > s.stop * (1 + 1e-12)) break;
        long g = llround(f / res);
        if (g % Q == 0) { g += g < (long)(M / 2) * Q ? 1 : -1; moved++; }  // off bin centers
        tones.push_back(g);
    }

    Sweeper sweeper(DefaultTaps(), Q, s.amp, s.scale);
    printf("pfb_response: M=%d T=%d (%s path), bin %d at %.6f MHz, fs %.6g Hz, channel %.6g Hz\n", M, T,
           pfb::UsesFused(M) ? "fused" : "separate", s.bin, s.bin * chan / 1e6, s.fs, chan);
    printf("  %zu tones, amplitude %g counts, %d rows per tone, grid %.6g Hz, tap gain %g\n", tones.size(), s.amp,
           Q, res, s.scale);
    if (moved) printf("  %d tone(s) on a bin center moved by one grid step\n", moved);
    printf("%12s  %13s  %12s  %12s  %9s  %9s  %12s\n", "freq_MHz", "offset_Hz", "aliased_Hz", "a_rms",
           "power_db", "gain_db", "predicted_db");

    FILE* csv = s.csv.empty() ? nullptr
        : OpenCsv(s.csv, "freq_hz,offset_from_center_hz,aliased_offset_hz,a_rms,power_db,gain_db,predicted_db");
    sweeper.Run(tones, 0, s.bin, [&](const Point& p) {
        const double f = p.grid * res, offset = p.offset * res, aliased = p.out_freq * chan;
        printf("%12.6f  %+13.1f  %+12.1f  %12.6g  %9.2f  %9.2f  %12.2f\n", f / 1e6, offset, aliased, p.a_rms,
               Db(p.a_rms), Db(p.gain), Db(p.predicted));
        if (csv)
            fprintf(csv, "%.3f,%.3f,%.3f,%.9g,%.4f,%.4f,%.4f\n", f, offset, aliased, p.a_rms, Db(p.a_rms),
                    Db(p.gain), Db(p.predicted));
    });
    if (csv) { fclose(csv); printf("wrote %s\n", s.csv.c_str()); }
    return 0;
}

static void Usage()
{
    fprintf(stderr,
            "usage: pfb_response [output directory]\n"
            "       pfb_response --bin K --start HZ --stop HZ --step HZ\n"
            "                    [--fs HZ] [--amp COUNTS] [--rows Q] [--scale S] [--csv FILE]\n"
            "  (see the comment at the top of tools/pfb_response.cpp)\n");
}

static int Main(int argc, char** argv)
{
    if (argc < 2 || argv[1][0] != '-') return Standard(argc > 1 ? argv[1] : ".");
    SweepArgs s;
    for (int i = 1; i < argc; i++) {
        const string opt = argv[i];
        if (opt == "-h" || opt == "--help") { Usage(); return 0; }
        if (i + 1 >= argc) { Usage(); throw invalid_argument(opt + " needs a value"); }
        const char* v = argv[++i];
        char* end = nullptr;
        const double x = strtod(v, &end);
        if (opt != "--csv" && (end == v || *end)) throw invalid_argument(opt + ": not a number: " + v);
        if (opt == "--bin") s.bin = (int)x;
        else if (opt == "--start") s.start = x;
        else if (opt == "--stop") s.stop = x;
        else if (opt == "--step") s.step = x;
        else if (opt == "--fs") s.fs = x;
        else if (opt == "--amp") s.amp = x;
        else if (opt == "--rows") s.rows = (int)x;
        else if (opt == "--scale") s.scale = x;
        else if (opt == "--csv") s.csv = v;
        else { Usage(); throw invalid_argument("unknown option " + opt); }
    }
    return Sweep(s);
}

// Library errors (bad arguments, CUDA or cuFFT failures) arrive as exceptions.
int main(int argc, char** argv)
{
    try {
        return Main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "pfb_response: %s\n", e.what());
        return 1;
    }
}

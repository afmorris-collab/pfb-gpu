// Device-only throughput of the PFB kernels (pfb.cuh), no PCIe transfers: input already on the
// GPU, output left on the GPU. For each M, 100 ms batches with T=32 taps per branch:
//   fused     pfb::Fused<M, T>     (M <= pfb::FUSED_MAX_M = 64)
//   separate  pfb::Separate<M, T>  (any M; FIR kernel + 4 cuFFT R2C)
// Times are per digitizer card (4 interleaved ADC channels); Gsps are for 3 digitizer cards =
// 12 ADC channels, processed one after another. Each path's DRAM traffic (the minimum it must
// move) is also given as a share of the GPU's nominal DRAM bandwidth, which bounds it. Each path's output is checked
// against the fp64 model (pfb_reference.hpp). Fused and separate alternate within every round,
// since GPU clocks drift and only same-run comparisons are meaningful. Throughput is the mean and
// standard deviation over 9 rounds, each round the mean of 20 batches; the time per card and DRAM
// traffic are at the mean throughput. Exit code 1 if any check fails.

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include "pfb/pfb.cuh"
#include "pfb_reference.hpp"
#include "bench.hpp"

constexpr int T = 32; // Number of filter taps per branch
static double peak_gbs;   // nominal DRAM bandwidth of this GPU in GB/s
static bool all_pass = true;

// Nominal DRAM bandwidth from the memory clock and bus width (double data rate).
static double PeakDramGBs()
{
    int dev, clock_khz, bus_bits;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    return 2.0 * clock_khz * 1e3 * (bus_bits / 8) / 1e9;
}

// Mean time of one call to run(), over 20 calls after a warm-up call.
template <typename F>
static float TimeMs(F run)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    run();
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < 20; i++) run();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms / 20;
}

template <int M>
static void RunM()
{
    constexpr bool kFused = pfb::UsesFused(M);
    const int Q = BatchRows(M);
    const size_t nin = pfb::InputSamples(M, T, Q), nout = pfb::OutputElements(M, Q);
    const std::vector<uint16_t> h_in = RandomInput(4 * nin);
    const std::vector<float> h_taps = RandomTaps(M, T);

    uint16_t* d_in; float* d_taps; float2* d_out; float* frames;
    CUDA_CHECK(cudaMalloc(&d_in, h_in.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_taps, h_taps.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, nout * sizeof(float2)));
    CUDA_CHECK(cudaMalloc(&frames, pfb::FrameElements(M, Q) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_taps, h_taps.data(), h_taps.size() * sizeof(float), cudaMemcpyHostToDevice));
    cufftHandle plan = pfb::SeparatePlan(M, Q);

    auto separate = [&] { pfb::Separate<M, T>(d_out, frames, d_in, d_taps, Q, plan); };
    // pfb::UsesFused(M), not kFused: nvcc 11.5 rejects a captured constexpr in if constexpr here.
    auto fused = [&] { if constexpr (pfb::UsesFused(M)) pfb::Fused<M, T>(d_out, d_in, d_taps, Q); };

    // Run once on a NaN-filled output and compare with the fp64 model on sampled rows.
    auto check = [&](auto run) {
        CUDA_CHECK(cudaMemset(d_out, 0xFF, nout * sizeof(float2)));
        run();
        std::vector<float2> out(nout);
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, nout * sizeof(float2), cudaMemcpyDeviceToHost));
        const PfbCheckResult res = PfbModelCheck(M, T, Q, h_in.data(), h_taps.data(), [&](int a, int b, int r) {
            const float2 v = out[((size_t)a * (M / 2 + 1) + b) * Q + r];
            return std::complex<float>(v.x, v.y);
        });
        all_pass &= res.pass;
        return res;
    };

    // Minimum DRAM traffic: input and output; the separate path also writes and reads the frames.
    const double in_bytes = nin * 4.0 * sizeof(uint16_t), out_bytes = nout * sizeof(float2);
    const double frame_bytes = pfb::FrameElements(M, Q) * sizeof(float);
    auto report = [&](const char* path, const std::vector<float>& times, double bytes, const PfbCheckResult& res) {
        std::vector<double> g;                 // Gsps of each round
        for (float t : times) g.push_back(Gsps(NCARDS * t, Q, M));
        const MeanSd s = Stats(g);
        const double t = BatchMs(s.mean, Q, M) / NCARDS, gbs = bytes / t / 1e6;
        printf("M=%4d %-9s %7.3f ms/card -> %6.2f +- %4.2f Gsps for %d ADC channels, DRAM %5.1f GB/s = "
               "%4.1f%% of nominal | vs fp64 model: %s (rel RMS %.1e)\n",
               M, path, t, s.mean, s.sd, NCH, gbs, 100.0 * gbs / peak_gbs,
               res.pass ? "PASS" : "FAIL", res.rel_rms_err);
    };

    const PfbCheckResult r_sep = check(separate);
    std::vector<float> ts, tf;
    if constexpr (kFused) {
        const PfbCheckResult r_fused = check(fused);
        for (int round = 0; round < 9; round++) { ts.push_back(TimeMs(separate)); tf.push_back(TimeMs(fused)); }
        report("fused", tf, in_bytes + out_bytes, r_fused);
    } else {
        for (int round = 0; round < 9; round++) ts.push_back(TimeMs(separate));
    }
    report("separate", ts, in_bytes + 2 * frame_bytes + out_bytes, r_sep);

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(frames));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_taps));
    CUDA_CHECK(cudaFree(d_out));
}

static int Main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    peak_gbs = PeakDramGBs();
    printf("%s, nominal DRAM bandwidth %.0f GB/s\n", prop.name, peak_gbs);
    printf("PFB kernels only (no PCIe), T=%d, 100 ms batches; %d digitizer cards x %d ADC channels\n"
           "DRAM GB/s = minimum DRAM traffic / time\n\n", T, NCARDS, pfb::NADC);
    RunM<8>(); RunM<16>(); RunM<32>(); RunM<50>(); RunM<64>();
    RunM<80>(); RunM<128>(); RunM<256>(); RunM<400>(); RunM<512>(); RunM<800>(); RunM<1024>();
    return all_pass ? 0 : 1;
}

// Library errors (bad arguments, CUDA or cuFFT failures) arrive as exceptions.
int main()
{
    try {
        return Main();
    } catch (const std::exception& e) {
        fprintf(stderr, "benchmarkpfb: %s\n", e.what());
        return 1;
    }
}

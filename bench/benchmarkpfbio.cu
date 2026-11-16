// End-to-end throughput of the operational PFB, including PCIe transfers, run the way the HAARP
// pfb service runs it: one Channelizer per digitizer card (copy in, filter bank, pitched copy of the
// selected bins back to pinned host memory). Uses the build's M and T (CMake PFB_M, PFB_T).
//
// For a 100 ms batch (Q rows), all bins or a quarter of them copied back, two ways of driving
// the digitizer cards:
//   serial:     process() for each digitizer card in turn (each waits for its copies)
//   overlapped: launch() for all digitizer cards, then wait() for all (as the pfb service does), so
//               one digitizer card's copies overlap another's kernel and copies on separate CUDA streams
// It reports the rate for all ADC channels and the PCIe traffic each way as a share of the link's
// nominal bandwidth (from its maximum generation and width, via NVML), which bounds it. It also
// checks that the selected bins arrive intact and the others are left untouched.

#include <cstdio>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <complex>
#include <memory>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>
#include <nvml.h>
#include "pfb/channelizer.hpp"
#include "pfb/pfb.cuh"
#include "pfb_reference.hpp"
#include "bench.hpp"

constexpr int M = PFB_M, T = PFB_T;
constexpr int NBINS = M / 2 + 1;
constexpr int PITCH_PAD = 4096;            // output rows are not contiguous in the ring buffer

using cplx = std::complex<float>;

// Nominal PCIe bandwidth per direction (GB/s) of the link to CUDA device 0, from its maximum
// generation and width; 0 if NVML cannot tell. Per lane: 8b/10b coding for gen 1-2, 128b/130b
// for gen 3-5, FLIT 242/256 for gen 6.
static double NominalPcieGBs(unsigned* gen_out, unsigned* width_out)
{
    static const double lane_gbs[] = {0, 0.25, 0.5, 0.985, 1.969, 3.938, 7.563};
    char bus_id[32];
    nvmlDevice_t dev;
    unsigned gen = 0, width = 0;
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), 0) != cudaSuccess || nvmlInit() != NVML_SUCCESS)
        return 0;
    const bool ok = nvmlDeviceGetHandleByPciBusId(bus_id, &dev) == NVML_SUCCESS &&
                    nvmlDeviceGetMaxPcieLinkGeneration(dev, &gen) == NVML_SUCCESS &&
                    nvmlDeviceGetMaxPcieLinkWidth(dev, &width) == NVML_SUCCESS;
    nvmlShutdown();
    if (!ok || gen < 1 || gen > 6) return 0;
    *gen_out = gen;
    *width_out = width;
    return lane_gbs[gen] * width;
}
static double pcie_gbs;                    // nominal PCIe bandwidth per direction

// Host buffers for one digitizer card, pinned the way the HAARP pfb service pins its ring buffers.
struct HostCard {
    std::vector<uint16_t> in;
    std::vector<cplx> out;                 // [adc][bin][pitch]
    pfb::PfbOutput view;
    HostCard(size_t nin, int Q) : in(RandomInput(4 * nin)), out((size_t)pfb::NADC * NBINS * (Q + PITCH_PAD)),
        view{out.data(), (size_t)Q + PITCH_PAD, (size_t)NBINS * (Q + PITCH_PAD)} {
        CUDA_CHECK(cudaHostRegister(in.data(), in.size() * sizeof(uint16_t), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(out.data(), out.size() * sizeof(cplx), cudaHostRegisterDefault));
    }
    ~HostCard() { cudaHostUnregister(in.data()); cudaHostUnregister(out.data()); }
    HostCard(const HostCard&) = delete;    // a copy would unregister the same memory twice
    HostCard& operator=(const HostCard&) = delete;
};

static void RunQ(int Q, const std::vector<float>& taps)
{
    std::vector<pfb::Channelizer> chans;
    std::vector<std::unique_ptr<HostCard>> host;
    for (int i = 0; i < NCARDS; i++) chans.emplace_back(M, T, Q, taps);
    for (int i = 0; i < NCARDS; i++) host.push_back(std::make_unique<HostCard>(chans[0].input_samples(), Q));
    auto serial = [&] {
        for (int i = 0; i < NCARDS; i++) chans[i].process(host[i]->in.data(), host[i]->view);
    };
    auto overlapped = [&] {
        for (int i = 0; i < NCARDS; i++) chans[i].launch(host[i]->in.data(), host[i]->view);
        for (auto& c : chans) c.wait();
    };

    // Reference: digitizer card 0's filter bank output, computed directly on the GPU.
    uint16_t* d_in; float* d_taps; float2* d_out;
    CUDA_CHECK(cudaMalloc(&d_in, host[0]->in.size() * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_taps, taps.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)pfb::NADC * NBINS * Q * sizeof(float2)));
    CUDA_CHECK(cudaMemcpy(d_in, host[0]->in.data(), host[0]->in.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_taps, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice));
    float* frames = nullptr;
    cufftHandle plan = 0;
    if (!pfb::UsesFused(M)) {
        CUDA_CHECK(cudaMalloc(&frames, pfb::FrameElements(M, Q) * sizeof(float)));
        plan = pfb::SeparatePlan(M, Q);
    }
    pfb::FilterBank<M, T>(d_out, frames, d_in, d_taps, Q, plan);
    if (!pfb::UsesFused(M)) cufftDestroy(plan);
    CUDA_CHECK(cudaFree(frames));
    std::vector<float2> ref((size_t)pfb::NADC * NBINS * Q);
    CUDA_CHECK(cudaMemcpy(ref.data(), d_out, ref.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_taps));
    CUDA_CHECK(cudaFree(d_out));

    printf("Q=%d rows per batch = %.0f ms of signal\n", Q, SignalMs(Q, M));
    PfbPrintCheck("filter bank vs fp64 model", PfbModelCheck(M, T, Q, host[0]->in.data(), taps.data(),
        [&](int a, int b, int r) { const float2 v = ref[((size_t)a * NBINS + b) * Q + r]; return cplx(v.x, v.y); }));
    for (int nbins : {NBINS, NBINS / 4}) {
        std::vector<int> bins(nbins);                          // a contiguous band in the middle
        std::iota(bins.begin(), bins.end(), (NBINS - nbins) / 2);
        for (auto& c : chans) c.set_bins(bins);
        const double in_bytes = (double)NCARDS * pfb::NADC * chans[0].input_samples() * sizeof(uint16_t);
        const double out_bytes = (double)NCARDS * pfb::NADC * nbins * Q * sizeof(cplx);

        for (int mode = 0; mode < 2; mode++) {
            auto run = [&] { if (mode == 0) serial(); else overlapped(); };

            // Correctness: digitizer card 0's selected bins equal the reference; the rest stay NaN.
            std::fill(host[0]->out.begin(), host[0]->out.end(), cplx(NAN, NAN));
            run();
            size_t bad = 0;
            for (int a = 0; a < pfb::NADC; a++)
                for (int b = 0; b < NBINS; b++) {
                    const bool selected = std::find(bins.begin(), bins.end(), b) != bins.end();
                    for (int r = 0; r < Q; r++) {
                        const cplx y = host[0]->out[a * host[0]->view.adc_stride + b * host[0]->view.bin_stride + r];
                        const float2 x = ref[((size_t)a * NBINS + b) * Q + r];
                        bad += selected ? (y.real() != x.x || y.imag() != x.y) : !std::isnan(y.real());
                    }
                }

            // Timing: median over 9 rounds of 5 batches, wall clock.
            std::vector<float> t;
            for (int round = 0; round < 45; round++) {
                const auto s = std::chrono::steady_clock::now();
                run();
                t.push_back(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - s).count());
            }
            const float tb = Median(t);
            const double in_gbs = in_bytes / tb / 1e6, out_gbs = out_bytes / tb / 1e6;
            printf("  %3d bins, %-12s %7.2f ms -> %5.2f Gsps for %d ADC channels, PCIe in %5.2f GB/s "
                   "(%4.1f%%), out %5.2f GB/s (%4.1f%%) | %s\n",
                   nbins, mode == 0 ? "serial:" : "overlapped:", tb, Gsps(tb, Q, M), NCH,
                   in_gbs, pcie_gbs ? 100 * in_gbs / pcie_gbs : 0.0, out_gbs,
                   pcie_gbs ? 100 * out_gbs / pcie_gbs : 0.0, bad ? "OUTPUT WRONG" : "output OK");
        }
    }
}

static int Main()
{
    printf("PFB M=%d T=%d (%s), %d digitizer cards x %d ADC channels at %.0f Msps = %.2f Gsps\n", M, T,
           pfb::UsesFused(M) ? "fused" : "separate", NCARDS, pfb::NADC, ADC_RATE / 1e6, NCH * ADC_RATE / 1e9);
    unsigned gen = 0, width = 0;
    pcie_gbs = NominalPcieGBs(&gen, &width);
    if (pcie_gbs)
        printf("PCIe gen %u x%u: nominal %.1f GB/s each way; percentages are of that\n\n", gen, width, pcie_gbs);
    else
        printf("PCIe link unknown (NVML unavailable); percentages shown as 0\n\n");
    RunQ(BatchRows(M), RandomTaps(M, T));
    return 0;
}

// Library errors (bad arguments, CUDA or cuFFT failures) arrive as exceptions.
int main()
{
    try {
        return Main();
    } catch (const std::exception& e) {
        fprintf(stderr, "benchmarkpfbio: %s\n", e.what());
        return 1;
    }
}

// Streaming throughput of the PFB including PCIe transfers, T=32, for M from 8 to 1024 (the same
// sweep as benchmarkpfb). For each of the 3 digitizer cards, a 100 ms batch is copied in from
// pinned host memory, filtered (pfb::FilterBank<M, T>: fused for M <= 64, else separate), and the
// selected bins are copied back to pinned host memory with one pitched copy per ADC channel, all
// on the card's own CUDA stream. Those are the steps of Channelizer::launch; Channelizer's M is
// fixed at build time, so this sweep has its own copy of them (StreamCard).
//
// Double buffered: two StreamCards per digitizer card, used for alternate batches, and batches
// queued back to back, so batch k+1 is copied in while batch k is still being filtered or copied
// back: the sustained rate of a continuous stream. Throughput: mean and standard deviation over 9
// rounds of PIPE_BATCHES batches each (filling and draining the pipeline is spread over the round);
// the time per batch and PCIe traffic are at the mean throughput.
// Two bin selections: all M/2+1 bins, and a quarter of them (a contiguous band in the middle).
//
// It reports Gsps for all 12 ADC channels and the PCIe traffic each way as a share of the link's
// nominal bandwidth (from its maximum generation and width, via NVML), which bounds it. It checks
// the filter bank output against the fp64 model (pfb_reference.hpp), and that the selected bins
// arrive intact in both host buffers and the other bins are left untouched. Exit code 1 if any
// check fails.

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>
#include "pfb/pfb.cuh"
#include "pfb_reference.hpp"
#include "bench.hpp"

constexpr int T = 32;                      // filter taps per branch
constexpr int NSLOTS = 2;                  // batches in flight per digitizer card
constexpr int PIPE_BATCHES = 20;           // batches per timed round
constexpr int PITCH_PAD = 4096;            // output rows are not contiguous in the ring buffer

using cplx = std::complex<float>;
static double pcie_gbs;                    // nominal PCIe bandwidth per direction
static bool all_pass = true;

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

// One batch in flight for one digitizer card: GPU buffers, a CUDA stream, and the pinned host
// output, laid out like the HAARP ring buffer: out[adc * adc_stride + bin * bin_stride + row].
template <int M>
struct StreamCard {
    static constexpr int NBINS = M / 2 + 1;
    const int Q;
    const size_t nin;                      // input samples per batch (4 x uint16 each)
    const size_t bin_stride, adc_stride;   // host output strides, in elements
    uint16_t* d_in = nullptr;
    float2* d_out = nullptr;               // filter bank output, [adc][bin][row]
    float* frames = nullptr;               // separate path only
    cufftHandle plan = 0;                  // separate path only
    cudaStream_t stream = nullptr;
    cudaEvent_t done = nullptr;
    std::vector<cplx> out;

    StreamCard(int Q_) : Q(Q_), nin(pfb::InputSamples(M, T, Q_)), bin_stride(Q_ + PITCH_PAD),
                         adc_stride(NBINS * bin_stride), out(pfb::NADC * adc_stride) {
        CUDA_CHECK(cudaMalloc(&d_in, pfb::NADC * nin * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_out, pfb::OutputElements(M, Q) * sizeof(float2)));
        if (!pfb::UsesFused(M)) {
            CUDA_CHECK(cudaMalloc(&frames, pfb::FrameElements(M, Q) * sizeof(float)));
            plan = pfb::SeparatePlan(M, Q);
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        CUDA_CHECK(cudaHostRegister(out.data(), out.size() * sizeof(cplx), cudaHostRegisterDefault));
    }
    ~StreamCard() {
        cudaHostUnregister(out.data());
        cudaEventDestroy(done);
        cudaStreamDestroy(stream);
        if (plan) cufftDestroy(plan);
        cudaFree(frames);
        cudaFree(d_out);
        cudaFree(d_in);
    }
    StreamCard(const StreamCard&) = delete;
    StreamCard& operator=(const StreamCard&) = delete;

    // Queue one batch: copy in, filter bank, copy bins first..first+nbins-1 back.
    void launch(const uint16_t* h_in, const float* d_taps, int first, int nbins) {
        CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, pfb::NADC * nin * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        pfb::FilterBank<M, T>(d_out, frames, d_in, d_taps, Q, plan, stream);
        const size_t row_bytes = Q * sizeof(cplx);
        for (int a = 0; a < pfb::NADC; a++)
            CUDA_CHECK(cudaMemcpy2DAsync(&out[a * adc_stride + first * bin_stride], bin_stride * sizeof(cplx),
                                         d_out + ((size_t)a * NBINS + first) * Q, row_bytes,
                                         row_bytes, nbins, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(done, stream));
    }
    // Wait for the batch queued by launch(); returns at once if none was.
    void wait() { CUDA_CHECK(cudaEventSynchronize(done)); }
};

template <int M>
static void RunM()
{
    using Card = StreamCard<M>;
    constexpr int NBINS = Card::NBINS;
    const int Q = BatchRows(M);
    const size_t nin = pfb::InputSamples(M, T, Q);
    const std::vector<float> h_taps = RandomTaps(M, T);
    std::vector<uint16_t> h_in = RandomInput(4 * nin);   // every card and batch reads this input
    CUDA_CHECK(cudaHostRegister(h_in.data(), h_in.size() * sizeof(uint16_t), cudaHostRegisterDefault));
    float* d_taps;
    CUDA_CHECK(cudaMalloc(&d_taps, h_taps.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_taps, h_taps.data(), h_taps.size() * sizeof(float), cudaMemcpyHostToDevice));

    // cards[slot * NCARDS + card]
    std::vector<std::unique_ptr<Card>> cards;
    for (int i = 0; i < NSLOTS * NCARDS; i++) cards.push_back(std::make_unique<Card>(Q));

    // nbatch batches back to back, each copying bins first..first+nbins-1 back.
    auto stream = [&](int nbatch, int first, int nbins) {
        for (int k = 0; k < nbatch; k++)
            for (int i = (k % NSLOTS) * NCARDS, end = i + NCARDS; i < end; i++) {
                cards[i]->wait();              // this slot's previous batch (k - NSLOTS) is done
                cards[i]->launch(h_in.data(), d_taps, first, nbins);
            }
        for (auto& c : cards) c->wait();
    };

    // Reference: one card's filter bank output, read straight from its GPU buffer, checked
    // against the fp64 model on sampled rows.
    stream(NSLOTS, 0, NBINS);
    std::vector<float2> ref(pfb::OutputElements(M, Q));
    CUDA_CHECK(cudaMemcpy(ref.data(), cards[0]->d_out, ref.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    const PfbCheckResult model = PfbModelCheck(M, T, Q, h_in.data(), h_taps.data(), [&](int a, int b, int r) {
        const float2 v = ref[((size_t)a * NBINS + b) * Q + r];
        return cplx(v.x, v.y);
    });
    all_pass &= model.pass;

    const double in_bytes = (double)NCARDS * pfb::NADC * nin * sizeof(uint16_t);
    for (int nbins : {NBINS, NBINS / 4}) {
        const int first = (NBINS - nbins) / 2;   // a contiguous band in the middle

        // Correctness: digitizer card 0's output in both slots: the selected bins equal the
        // reference, the rest stay NaN.
        for (int s = 0; s < NSLOTS; s++) std::fill(cards[s * NCARDS]->out.begin(), cards[s * NCARDS]->out.end(), cplx(NAN, NAN));
        stream(PIPE_BATCHES, first, nbins);
        size_t bad = 0;
        for (int s = 0; s < NSLOTS; s++) {
            const Card& c = *cards[s * NCARDS];
            for (int a = 0; a < pfb::NADC; a++)
                for (int b = 0; b < NBINS; b++) {
                    const bool selected = b >= first && b < first + nbins;
                    for (int r = 0; r < Q; r++) {
                        const cplx y = c.out[a * c.adc_stride + b * c.bin_stride + r];
                        const float2 x = ref[((size_t)a * NBINS + b) * Q + r];
                        bad += selected ? (y.real() != x.x || y.imag() != x.y) : !std::isnan(y.real());
                    }
                }
        }
        all_pass &= bad == 0;

        std::vector<double> g;                 // Gsps of each round
        for (int round = 0; round < 9; round++) {
            const auto s = std::chrono::steady_clock::now();
            stream(PIPE_BATCHES, first, nbins);
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
            g.push_back(Gsps(ms / PIPE_BATCHES, Q, M));
        }
        const MeanSd gs = Stats(g);
        const double tb = BatchMs(gs.mean, Q, M);
        const double out_bytes = (double)NCARDS * pfb::NADC * nbins * Q * sizeof(cplx);
        const double in_gbs = in_bytes / tb / 1e6, out_gbs = out_bytes / tb / 1e6;
        printf("M=%4d %-8s %3d of %3d bins: %7.2f ms -> %6.2f +- %4.2f Gsps for %d ADC channels, PCIe in %5.2f GB/s "
               "(%4.1f%%), out %5.2f GB/s (%4.1f%%) | vs fp64 model: %s, output %s\n",
               M, pfb::UsesFused(M) ? "fused" : "separate", nbins, NBINS, tb, gs.mean, gs.sd, NCH,
               in_gbs, pcie_gbs ? 100 * in_gbs / pcie_gbs : 0.0, out_gbs, pcie_gbs ? 100 * out_gbs / pcie_gbs : 0.0,
               model.pass ? "PASS" : "FAIL", bad ? "WRONG" : "OK");
    }

    cards.clear();
    CUDA_CHECK(cudaFree(d_taps));
    CUDA_CHECK(cudaHostUnregister(h_in.data()));
}

static int Main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    unsigned gen = 0, width = 0;
    pcie_gbs = NominalPcieGBs(&gen, &width);
    printf("%s, ", prop.name);
    if (pcie_gbs)
        printf("PCIe gen %u x%u: nominal %.1f GB/s each way; percentages are of that\n", gen, width, pcie_gbs);
    else
        printf("PCIe link unknown (NVML unavailable); percentages shown as 0\n");
    printf("Streaming PFB incl. PCIe, double buffered, T=%d, 100 ms batches; %d digitizer cards x %d ADC "
           "channels at %.0f Msps = %.2f Gsps\n\n", T, NCARDS, pfb::NADC, ADC_RATE / 1e6, NCH * ADC_RATE / 1e9);
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
        fprintf(stderr, "benchmarkpfbio: %s\n", e.what());
        return 1;
    }
}

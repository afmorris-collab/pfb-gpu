// Channelizer: the polyphase filterbank (pfb.cuh) for one digitizer card, with host copies, for
// operational use. M and T are fixed at build time (CMake PFB_M, PFB_T).
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
#include "sizes.hpp"

namespace pfb {

// Where one batch of output goes: element (adc, bin, row) is
//     data[adc * adc_stride + bin * bin_stride + row]       (strides in elements)
// Rows are contiguous, so both strides must be at least Q. Only the bins selected by
// Channelizer::set_bins() are written; the other bin slots are left untouched. The strides must
// not make two (adc, bin) slots overlap (e.g. both equal to Q); that is not checked, and the
// overlapping slots would overwrite each other. Use pinned memory (cudaHostRegister or
// cudaMallocHost), or the copies are not asynchronous.
struct PfbOutput {
    std::complex<float>* data;
    size_t bin_stride;
    size_t adc_stride;
};

// Polyphase filterbank for one digitizer card: 4 interleaved int16 ADC channels, 2x oversampled,
// M/2+1 output bins. Owns its device buffers and a CUDA stream, so several Channelizers can
// overlap their copies and kernels. Not copyable; movable (a moved-from Channelizer holds nothing
// and may only be destroyed or assigned to). Errors throw std::invalid_argument (bad arguments)
// or std::runtime_error (CUDA/cuFFT failures).
class Channelizer {
public:
    // M, T: must equal the build's PFB_M, PFB_T. Q: output rows per batch, a multiple of
    // QMultiple(M) (8 for the fused path, UsesFused(M); else 10). taps: T*M prototype filter
    // (e.g. KaiserTaps).
    Channelizer(int M, int T, int Q, const std::vector<float>& taps);
    ~Channelizer();
    Channelizer(const Channelizer&) = delete;
    Channelizer& operator=(const Channelizer&) = delete;
    Channelizer(Channelizer&& other) noexcept;
    Channelizer& operator=(Channelizer&& other) noexcept;

    // Bins to copy back, in any order, each in 0..M/2. Default: all bins.
    void set_bins(const std::vector<int>& bins);

    // Queue one batch on this Channelizer's stream and return without waiting: copy
    // input_samples() interleaved samples (4 x uint16 offset binary each, contiguous) in, run
    // the filter bank, and copy the selected bins back to out. input and out must stay valid
    // until wait() returns.
    void launch(const uint16_t* input, const PfbOutput& out);
    // Wait for the batch queued by launch().
    void wait();
    // launch() followed by wait().
    void process(const uint16_t* input, const PfbOutput& out);

    size_t input_samples() const;   // (Q - 1 + 2T) * M/2: input samples read per batch
    size_t input_advance() const;   // Q * M/2: input samples between the starts of successive batches
    int M() const { return M_; }
    int T() const { return T_; }
    int Q() const { return Q_; }
    int nbins() const { return M_ / 2 + 1; }

private:
    void release() noexcept;        // free everything owned; leaves the object empty
    void steal(Channelizer& other) noexcept;

    int M_ = 0, T_ = 0, Q_ = 0;
    std::vector<int> bins_;
    uint16_t* d_in_ = nullptr;                // input, 4 x input_samples() uint16
    float* d_taps_ = nullptr;
    std::complex<float>* d_out_ = nullptr;    // output, [adc][bin][row]
    float* d_frames_ = nullptr;               // separate path: FIR output / FFT input
    cufftHandle plan_ = 0;                    // separate path
    bool has_plan_ = false;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t event_ = nullptr;
};

}  // namespace pfb

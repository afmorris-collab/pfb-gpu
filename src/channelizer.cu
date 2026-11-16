// Channelizer (channelizer.hpp): one digitizer card's PFB with host copies.

#include <stdexcept>
#include <string>
#include <utility>
#include <cuda_runtime.h>
#include <cufft.h>

#include "pfb/channelizer.hpp"
#include "pfb/cuda_check.hpp"
#include "pfb/pfb.cuh"

namespace pfb {

// PFB size is fixed at build time (CMake PFB_M, PFB_T).
constexpr bool kFused = UsesFused(PFB_M);

Channelizer::Channelizer(int M, int T, int Q, const std::vector<float>& taps) : M_(M), T_(T), Q_(Q) {
    if (M != PFB_M || T != PFB_T)
        throw std::invalid_argument("pfb::Channelizer: M=" + std::to_string(M) + ", T=" + std::to_string(T) +
                                    " does not match the build (PFB_M=" + std::to_string(PFB_M) +
                                    ", PFB_T=" + std::to_string(PFB_T) + ")");
    if (taps.size() != (size_t)T * M)
        throw std::invalid_argument("pfb::Channelizer: expected T*M = " + std::to_string(T * M) +
                                    " taps, got " + std::to_string(taps.size()));
    const int q_multiple = QMultiple(M);
    if (Q <= 0 || Q % q_multiple)
        throw std::invalid_argument("pfb::Channelizer: Q must be a positive multiple of " +
                                    std::to_string(q_multiple));
    if (!kFused && !SeparateFitsInt(M, T, Q))
        throw std::invalid_argument("pfb::Channelizer: Q too large (4*Q*M must be below 2^31)");

    bins_.resize(nbins());
    for (int b = 0; b < nbins(); b++) bins_[b] = b;

    try {
        PFB_CUDA_CHECK(cudaMalloc(&d_in_, NADC * input_samples() * sizeof(uint16_t)));
        PFB_CUDA_CHECK(cudaMalloc(&d_taps_, taps.size() * sizeof(float)));
        PFB_CUDA_CHECK(cudaMemcpy(d_taps_, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice));
        PFB_CUDA_CHECK(cudaMalloc(&d_out_, OutputElements(M, Q) * sizeof(std::complex<float>)));
        if (!kFused) {
            PFB_CUDA_CHECK(cudaMalloc(&d_frames_, FrameElements(M, Q) * sizeof(float)));
            plan_ = SeparatePlan(M, Q);
            has_plan_ = true;
        }
        PFB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        PFB_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventBlockingSync));
    } catch (...) {
        release();
        throw;
    }
}

Channelizer::~Channelizer() { release(); }

Channelizer::Channelizer(Channelizer&& other) noexcept { steal(other); }

Channelizer& Channelizer::operator=(Channelizer&& other) noexcept {
    if (this != &other) {
        release();
        steal(other);
    }
    return *this;
}

void Channelizer::release() noexcept {
    if (has_plan_) cufftDestroy(plan_);
    if (event_) cudaEventDestroy(event_);
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(d_in_);                          // cudaFree(nullptr) is a no-op
    cudaFree(d_taps_);
    cudaFree(d_out_);
    cudaFree(d_frames_);
    d_in_ = nullptr; d_taps_ = nullptr; d_out_ = nullptr; d_frames_ = nullptr;
    has_plan_ = false; event_ = nullptr; stream_ = nullptr;
}

void Channelizer::steal(Channelizer& other) noexcept {
    M_ = other.M_; T_ = other.T_; Q_ = other.Q_;
    bins_ = std::move(other.bins_);
    d_in_ = std::exchange(other.d_in_, nullptr);
    d_taps_ = std::exchange(other.d_taps_, nullptr);
    d_out_ = std::exchange(other.d_out_, nullptr);
    d_frames_ = std::exchange(other.d_frames_, nullptr);
    plan_ = other.plan_;
    has_plan_ = std::exchange(other.has_plan_, false);
    stream_ = std::exchange(other.stream_, nullptr);
    event_ = std::exchange(other.event_, nullptr);
}

void Channelizer::set_bins(const std::vector<int>& bins) {
    for (int b : bins)
        if (b < 0 || b >= nbins())
            throw std::invalid_argument("pfb::Channelizer::set_bins: bin " + std::to_string(b) +
                                        " outside 0.." + std::to_string(nbins() - 1));
    bins_ = bins;
}

void Channelizer::launch(const uint16_t* input, const PfbOutput& out) {
    if (!out.data || out.bin_stride < (size_t)Q_ || out.adc_stride < (size_t)Q_)
        throw std::invalid_argument("pfb::Channelizer::launch: output needs data and strides >= Q");

    // --------------------------------- Copy input to the GPU -------------------------
    PFB_CUDA_CHECK(cudaMemcpyAsync(d_in_, input, NADC * input_samples() * sizeof(uint16_t),
                                   cudaMemcpyHostToDevice, stream_));

    // --------------------------------- Filter bank: FIR + FFT ------------------------
    FilterBank<PFB_M, PFB_T>(reinterpret_cast<float2*>(d_out_), d_frames_, d_in_, d_taps_, Q_, plan_,
                             stream_);

    // --------------------------------- Copy selected bins back -----------------------
    // Each bin is Q contiguous samples on the GPU. A run of consecutive bins goes back as one
    // pitched copy per ADC channel (all bins: 4 copies in total).
    const size_t row_bytes = sizeof(std::complex<float>) * Q_;
    for (size_t i = 0; i < bins_.size(); ) {
        size_t n = 1;
        while (i + n < bins_.size() && bins_[i + n] == bins_[i] + (int)n) n++;
        const int bin = bins_[i];
        for (int a = 0; a < NADC; a++) {
            PFB_CUDA_CHECK(cudaMemcpy2DAsync(out.data + a * out.adc_stride + bin * out.bin_stride,
                                             out.bin_stride * sizeof(std::complex<float>),
                                             d_out_ + ((size_t)a * nbins() + bin) * Q_, row_bytes,
                                             row_bytes, n, cudaMemcpyDeviceToHost, stream_));
        }
        i += n;
    }
    PFB_CUDA_CHECK(cudaEventRecord(event_, stream_));
}

void Channelizer::wait() {
    PFB_CUDA_CHECK(cudaEventSynchronize(event_));
}

void Channelizer::process(const uint16_t* input, const PfbOutput& out) {
    launch(input, out);
    wait();
}

size_t Channelizer::input_samples() const { return InputSamples(M_, T_, Q_); }
size_t Channelizer::input_advance() const { return (size_t)Q_ * (M_ / 2); }

}  // namespace pfb

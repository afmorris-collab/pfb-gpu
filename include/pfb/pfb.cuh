// Polyphase filterbank (PFB) for 4 interleaved int16 ADC channels, 2x oversampled: the low-level
// GPU API. Channelizer (channelizer.hpp) wraps it with host copies for operational use.
//
// Input:  rows of M/2 samples, one ushort4 (4 ADC channels, offset binary) per sample;
//         InputSamples(M, T, Q) samples for Q output rows.
// Taps:   T*M prototype filter, stored as 2T rows of M/2.
// Output: out[adc][bin][row], complex, M/2+1 bins per row; OutputElements(M, Q) in all.
//
// Output row r is the FFT of the M-sample frame
//     y[r][c]       = sum_{k even} taps[k][c] * x[r+k][c]     c = 0..M/2-1
//     y[r][c + M/2] = sum_{k odd}  taps[k][c] * x[r+k][c]     k = 0..2T-1
// with the two halves swapped on odd rows (frames start M/2 samples apart).
//
// Two implementations, and FilterBank<M, T>, which picks between them:
//   Fused<M, T>     M <= FUSED_MAX_M (64): FIR and FFT in one kernel (cuFFTDx Thread() FFT)
//   Separate<M, T>  any M:   FirKernel (fir_kernel.cuh) into a frame buffer, then 4 cuFFT R2C calls
// All errors are reported by throwing std::invalid_argument or std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftdx.hpp>
#include "cuda_check.hpp"
#include "fir_kernel.cuh"
#include "sizes.hpp"

// cuFFTDx target architecture (e.g. 860 for sm_86), set by CMake from CMAKE_CUDA_ARCHITECTURES.
#ifndef PFB_CUFFTDX_SM
#error "PFB_CUFFTDX_SM must be defined (e.g. 860 for sm_86)"
#endif

namespace pfb {

// Sizes, limits and UsesFused(M) are in sizes.hpp.

// ---------------------------------------------------------------------------------------------
// Fused kernel: one block = 8 output rows x 4 channels, 32 threads.
//   1. FIR: thread c (< M/2) computes column c of the 8 frames for all 4 channels. Each input
//      row is loaded once and added into every output row whose 2T-row window contains it.
//   2. The frames are passed through shared memory (pitch M+1: conflict-free row reads).
//   3. FFT: thread = (channel, row); cuFFTDx Thread() FFT in registers, then each thread writes
//      its M/2+1 bins. Consecutive threads hold consecutive rows, so the stores coalesce.
// ---------------------------------------------------------------------------------------------
template <int M, int T>
__global__ void __launch_bounds__(NADC * FUSED_ROWS)
FusedKernel(float2* __restrict__ out, const ushort4* __restrict__ in,
            const float* __restrict__ taps, int Q)
{
    using FFT = decltype(cufftdx::Size<M>() + cufftdx::Type<cufftdx::fft_type::r2c>()
                         + cufftdx::Direction<cufftdx::fft_direction::forward>()
                         + cufftdx::Precision<float>()
                         + cufftdx::RealFFTOptions<cufftdx::complex_layout::natural,
                                                   cufftdx::real_mode::normal>()
                         + cufftdx::SM<PFB_CUFFTDX_SM>() + cufftdx::Thread());
    constexpr int R = FUSED_ROWS;
    constexpr int K = 2 * T;                  // tap rows
    __shared__ float frame[NADC][R][M + 1];
    const size_t row0 = (size_t)blockIdx.x * R;

    // 1. FIR
    const int c = threadIdx.x;
    if (c < M / 2) {
        float h[K];
        #pragma unroll
        for (int k = 0; k < K; k++) h[k] = taps[k * (M / 2) + c];

        float acc[NADC][R][2] = {};           // [channel][output row][frame half]
        #pragma unroll
        for (int j = 0; j < R + K - 1; j++) {
            const ushort4 v = in[(row0 + j) * (M / 2) + c];
            const float x[NADC] = {v.x - 32768.f, v.y - 32768.f, v.z - 32768.f, v.w - 32768.f};
            #pragma unroll
            for (int i = 0; i < R; i++) {
                const int k = j - i;          // tap row that input row j meets in output row i
                if (k < 0 || k >= K) continue;
                #pragma unroll
                for (int a = 0; a < NADC; a++)
                    acc[a][i][k & 1] = fmaf(h[k], x[a], acc[a][i][k & 1]);
            }
        }
        // row0 is even, so output row i is odd exactly when i is: swap the halves there.
        #pragma unroll
        for (int i = 0; i < R; i++)
            #pragma unroll
            for (int a = 0; a < NADC; a++) {
                frame[a][i][c]         = acc[a][i][i & 1];
                frame[a][i][c + M / 2] = acc[a][i][(i & 1) ^ 1];
            }
    }
    __syncthreads();

    // 3. FFT
    using complex_type = typename FFT::value_type;
    const int ch = threadIdx.x / R, row = threadIdx.x % R;
    complex_type d[FFT::storage_size];
    #pragma unroll
    for (int n = 0; n < M; n++) reinterpret_cast<float*>(d)[n] = frame[ch][row][n];
    FFT().execute(d);
    complex_type* dst = reinterpret_cast<complex_type*>(out) + (size_t)ch * (M / 2 + 1) * Q + row0 + row;
    #pragma unroll
    for (int bin = 0; bin < M / 2 + 1; bin++) dst[(size_t)bin * Q] = d[bin];
}

// Fused FIR + FFT. M even, M <= FUSED_MAX_M; any T. Q a multiple of FUSED_Q_MULTIPLE.
//   out: OutputElements(M, Q) complex; in: InputSamples(M, T, Q) ushort4; taps: T*M floats.
template <int M, int T>
void Fused(float2* out, const uint16_t* in, const float* taps, int Q, cudaStream_t stream = 0)
{
    static_assert(M % 2 == 0 && UsesFused(M), "fused PFB supports even M <= FUSED_MAX_M (cuFFTDx Thread() FFT)");
    if (Q <= 0 || Q % FUSED_Q_MULTIPLE)
        throw std::invalid_argument("pfb::Fused: Q must be a positive multiple of 8");
    FusedKernel<M, T><<<Q / FUSED_ROWS, NADC * FUSED_ROWS, 0, stream>>>(
        out, reinterpret_cast<const ushort4*>(in), taps, Q);
    PFB_CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------
// Separate kernels: FirKernel into a frame buffer (4 ADC channels, Q frames of M floats each),
// then cuFFT.
// ---------------------------------------------------------------------------------------------

// cuFFT plan for Separate: Q real FFTs of length M, output transposed to out[bin][row].
// The caller owns the plan (cufftDestroy).
inline cufftHandle SeparatePlan(int M, int Q)
{
    cufftHandle plan;
    int n[1] = {M}, inembed[1] = {M}, onembed[1] = {M / 2 + 1};
    PFB_CUFFT_CHECK(cufftPlanMany(&plan, 1, n, inembed, 1, M, onembed, Q, 1, CUFFT_R2C, Q));
    return plan;
}

// Separate FIR and FFT. Any even M; T even, and a multiple of 8 if T > 20. Q a multiple of
// SEPARATE_Q_MULTIPLE, and small enough for SeparateFitsInt.
//   out:    OutputElements(M, Q) complex
//   frames: FrameElements(M, Q) floats of scratch: the FIR output, which the FFT reads
//   in:     InputSamples(M, T, Q) ushort4; taps: T*M floats
//   plan:   SeparatePlan(M, Q); its stream is set to `stream`
template <int M, int T>
void Separate(float2* out, float* frames, const uint16_t* in, const float* taps,
              int Q, cufftHandle plan, cudaStream_t stream = 0)
{
    static_assert(M % 2 == 0, "PFB needs even M");
    constexpr int NITER = SEPARATE_Q_MULTIPLE;  // output rows per thread
    // Tap passes: one if the whole 2T-row window fits in registers (<= 40 rows), else passes of
    // 16 rows (the per-pass accumulators then also need registers).
    constexpr int NPASSES = 2 * T <= 40 ? 1 : 2 * T / 16;
    static_assert((2 * T) % NPASSES == 0, "2T must split evenly into tap passes");
    if (Q <= 0 || Q % NITER)
        throw std::invalid_argument("pfb::Separate: Q must be a positive multiple of 10");
    if (!SeparateFitsInt(M, T, Q))
        throw std::invalid_argument("pfb::Separate: Q too large (4*Q*M must be below 2^31)");
    int nchunk = 1;                           // split the M/2 columns evenly into blocks of <= 64 threads
    while (M / 2 / nchunk > 64 || (M / 2) % nchunk) nchunk++;
    FirKernel<T, NITER, NPASSES><<<dim3(Q / NITER, nchunk), M / 2 / nchunk, 0, stream>>>(
        frames, Q * M, in, taps);
    PFB_CUDA_CHECK(cudaGetLastError());
    PFB_CUFFT_CHECK(cufftSetStream(plan, stream));
    for (int a = 0; a < NADC; a++)
        PFB_CUFFT_CHECK(cufftExecR2C(plan, frames + (size_t)a * Q * M,
                                     reinterpret_cast<cufftComplex*>(out + (size_t)a * (M / 2 + 1) * Q)));
}

// Fused<M, T> if UsesFused(M), else Separate<M, T>. The fused path ignores frames and plan (they
// may be null / 0).
template <int M, int T>
void FilterBank(float2* out, float* frames, const uint16_t* in, const float* taps,
                int Q, cufftHandle plan, cudaStream_t stream = 0)
{
    if constexpr (UsesFused(M))
        Fused<M, T>(out, in, taps, Q, stream);
    else
        Separate<M, T>(out, frames, in, taps, Q, plan, stream);
}

}  // namespace pfb

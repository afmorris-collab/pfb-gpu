// Sizes and limits of the PFB, shared by the GPU code (pfb.cuh) and host code (channelizer.hpp,
// tests): plain C++, no CUDA needed.
#pragma once

#include <climits>
#include <cstddef>

namespace pfb {

constexpr int NADC = 4;                    // interleaved ADC channels per digitizer card

// M <= FUSED_MAX_M uses the fused kernel (its cuFFTDx thread FFT holds a whole M-point FFT in one
// thread's registers); larger M the separate FIR and cuFFT kernels.
constexpr int FUSED_MAX_M = 64;
constexpr bool UsesFused(int M) { return M <= FUSED_MAX_M; }

constexpr int FUSED_ROWS = 8;              // output rows per block in the fused kernel
constexpr int FUSED_Q_MULTIPLE = FUSED_ROWS;
constexpr int SEPARATE_Q_MULTIPLE = 10;    // output rows per thread in the separate FIR kernel
constexpr int QMultiple(int M) { return UsesFused(M) ? FUSED_Q_MULTIPLE : SEPARATE_Q_MULTIPLE; }

// Buffer sizes in elements for Q output rows.
constexpr size_t InputSamples(int M, int T, int Q) { return (size_t)(Q - 1 + 2 * T) * (M / 2); }  // 4 x uint16
constexpr size_t OutputElements(int M, int Q) { return (size_t)NADC * (M / 2 + 1) * Q; }        // complex
constexpr size_t FrameElements(int M, int Q) { return (size_t)NADC * Q * M; }                  // float

// The separate path's FIR kernel indexes its input and frame buffer with 32-bit ints, which
// limits a batch to 4*Q*M < 2^31 (about 3.3 s of signal at 80 Msps).
constexpr bool SeparateFitsInt(int M, int T, int Q)
{
    return FrameElements(M, Q) <= INT_MAX && InputSamples(M, T, Q) <= INT_MAX;
}

}  // namespace pfb

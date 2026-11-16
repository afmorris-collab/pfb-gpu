// FIR stage of the separate PFB path (see pfb::Separate in pfb.cuh): computes the filtered
// M-sample frames that cuFFT then transforms.
//
// Thread layout: blockIdx.x selects NITER consecutive output rows, and blockIdx.y together with
// threadIdx.x selects one column c of the M/2 input columns. Each thread computes column c of its
// NITER frames for all 4 ADC channels:
//     frame[r][c]       = sum_{k even} taps[k][c] * x[r+k][c]      k = 0..2T-1
//     frame[r][c + M/2] = sum_{k odd}  taps[k][c] * x[r+k][c]
// with the two halves swapped on odd rows r.
//
// The window of input rows is kept in registers as a circular buffer: after output row i, the
// oldest input row (slot i % DEPTH) is replaced by the next one, so every input row is loaded
// once. All loops are fully unrolled, so every circular index is a compile-time constant and maps
// straight to a register.
//
// If the whole window does not fit in registers (2T > 40), the taps are processed in NPASSES
// passes of DEPTH = 2T/NPASSES rows each, and per-row sums are carried between passes.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pfb {

// Input row r, column c, for the 4 ADC channels: offset binary converted to float.
__device__ __forceinline__ float4 LoadSample(const uint16_t* in, int r, int c, int M)
{
    const ushort4 v = reinterpret_cast<const ushort4*>(in)[r * (M / 2) + c];
    return make_float4((float)((int32_t)v.x - 32768), (float)((int32_t)v.y - 32768),
                       (float)((int32_t)v.z - 32768), (float)((int32_t)v.w - 32768));
}

// Where the even-tap sum (.x) and odd-tap sum (.y) of output row r, column c go in one ADC
// channel's frame buffer: the first and second half of frame r, swapped on odd rows.
__device__ __forceinline__ int2 FrameIndex(int r, int c, int M)
{
    const int first = r * M + c, second = first + M / 2;
    return (r & 1) ? make_int2(second, first) : make_int2(first, second);
}

template <int T, int NITER, int NPASSES>
__global__ void FirKernel(
    float* __restrict__ frames,         // 4 ADC channels, adc_stride floats apart
    int adc_stride,
    const uint16_t* __restrict__ in,
    const float* __restrict__ taps)
{
    constexpr int DEPTH = 2 * T / NPASSES;    // tap rows per pass
    static_assert(DEPTH % 4 == 0, "2T/NPASSES must be a multiple of 4 (the tap loop takes 4 per step)");

    const int M = 2 * blockDim.x * gridDim.y;
    const int c = blockIdx.y * (M / 2 / gridDim.y) + threadIdx.x;   // M/2/gridDim.y == blockDim.x
    const int row0 = NITER * blockIdx.x;      // first output row of this block

    float total[4][2][NITER] = {};            // multi-pass only: [adc][even/odd taps][row]

    for (int pass = 0; pass < NPASSES; pass++) {
        // Load this pass's taps and the first DEPTH input rows of the window.
        float h[DEPTH];
        float x[4][DEPTH];                    // [adc][circular slot]
        #pragma unroll
        for (int j = 0; j < DEPTH; j++) h[j] = taps[(pass * DEPTH + j) * M / 2 + c];
        #pragma unroll
        for (int j = 0; j < DEPTH; j++) {
            const float4 v = LoadSample(in, row0 + pass * DEPTH + j, c, M);
            x[0][j] = v.x; x[1][j] = v.y; x[2][j] = v.z; x[3][j] = v.w;
        }

        #pragma unroll
        for (int i = 0; i < NITER; i++) {
            // Output row row0 + i: input row i + j sits in slot (i + j) % DEPTH.
            float sum[4][2] = {};             // [adc][even/odd taps]
            #pragma unroll
            for (int j = 0; j < DEPTH; j += 4) {
                #pragma unroll
                for (int a = 0; a < 4; a++) {
                    sum[a][0] += h[j]     * x[a][(i + j    ) % DEPTH]
                               + h[j + 2] * x[a][(i + j + 2) % DEPTH];
                    sum[a][1] += h[j + 1] * x[a][(i + j + 1) % DEPTH]
                               + h[j + 3] * x[a][(i + j + 3) % DEPTH];
                }
            }

            if constexpr (NPASSES > 1) {
                #pragma unroll
                for (int a = 0; a < 4; a++) {
                    total[a][0][i] += sum[a][0];
                    total[a][1][i] += sum[a][1];
                }
            } else {
                const int2 o = FrameIndex(row0 + i, c, M);
                #pragma unroll
                for (int a = 0; a < 4; a++) {
                    frames[a * adc_stride + o.x] = sum[a][0];
                    frames[a * adc_stride + o.y] = sum[a][1];
                }
            }

            // Replace the oldest slot with the next input row. Skipped after the last row: that
            // row is never used (the next pass reloads the window) and for the last block it lies
            // past the end of the input.
            if (i < NITER - 1) {
                const float4 v = LoadSample(in, row0 + pass * DEPTH + DEPTH + i, c, M);
                const int s = i % DEPTH;
                x[0][s] = v.x; x[1][s] = v.y; x[2][s] = v.z; x[3][s] = v.w;
            }
        }
    }

    if constexpr (NPASSES > 1) {
        // Write the sums one ADC channel at a time, which keeps a single output pointer live (the
        // sums already fill the register file).
        #pragma unroll
        for (int a = 0; a < 4; a++) {
            float* out = frames + (size_t)a * adc_stride;
            #pragma unroll
            for (int i = 0; i < NITER; i++) {
                const int2 o = FrameIndex(row0 + i, c, M);
                out[o.x] = total[a][0][i];
                out[o.y] = total[a][1][i];
            }
        }
    }
}

}  // namespace pfb

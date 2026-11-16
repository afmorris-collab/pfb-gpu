// Shared by the benchmarks and test_pfb: the system model, throughput in Gsps for all ADC
// channels, batch sizing, random test data and CUDA_CHECK (exits on error).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>

// System model (HAARP receiver): 3 digitizer cards, each 4 interleaved ADC channels at 80 Msps
// (12 x 80 Msps = 0.96 Gsps in all). Gsps figures are for all NCH ADC channels.
constexpr int NCARDS = 3;                  // digitizer cards (input streams, one Channelizer each)
constexpr int NCH = NCARDS * 4;            // ADC channels in total (pfb::NADC = 4 per digitizer card)
constexpr double ADC_RATE = 80e6;          // samples/s per ADC channel

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// Rows per batch holding about `seconds` of signal, a multiple of 40 (8 for the fused path,
// 10 for the separate one). Batches of fixed duration keep memory use similar at any M.
inline int BatchRows(int M, double seconds = 100e-3) { return (int)(seconds * ADC_RATE / (M / 2)) / 40 * 40; }

// Signal time in a batch of Q rows, and the throughput when all NCARDS digitizer cards take t_ms.
inline double SignalMs(int Q, int M) { return 1e3 * Q * (M / 2) / ADC_RATE; }
inline double Gsps(double t_ms, int Q, int M) { return (double)Q * (M / 2) * NCH / t_ms / 1e6; }

inline float Median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// Random offset-binary samples, and random taps (T*M) with the filter gain kept near 1. Fixed
// seeds and raw std::mt19937 output (whose sequence the C++ standard specifies, unlike the
// std::*_distribution classes), so every platform gets the same data.
inline std::vector<uint16_t> RandomInput(size_t n, uint32_t seed = 1)
{
    std::mt19937 gen(seed);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = gen() & 0xFFFF;
    return v;
}
inline std::vector<float> RandomTaps(int M, int T, uint32_t seed = 2)
{
    std::mt19937 gen(seed);
    std::vector<float> v((size_t)T * M);
    for (auto& x : v) x = (float)(gen() / 4294967296.0 / T);
    return v;
}

# pfb-gpu

GPU polyphase filterbank (PFB) for 4 interleaved int16 ADC channels, 2x oversampled, producing
M/2+1 complex bins per output row. Two implementations, chosen at build time from M:

- **fused** (M ≤ 64): FIR and FFT in one kernel; the FFT is a cuFFTDx thread FFT, so the
  filtered frames never go to GPU memory.
- **separate** (any even M): a register-resident FIR kernel writes the frames to GPU memory,
  then cuFFT R2C transforms them.

This code was written for a receiver at HAARP (the High-frequency Active Auroral Research
Program observatory in Gakona, Alaska), which digitizes 12 ADC channels at 80 Msps on three
digitizer cards of 4 channels each. The benchmarks report throughput against that system.

## How it works

The input is one digitizer card's data: a stream of samples, each holding 4 offset-binary
`uint16` values, one per ADC channel (value − 32768 is the signed sample). The filter bank splits
each ADC channel into M/2+1 frequency bins spaced fs/M apart (fs = sample rate), from DC (bin 0)
to Nyquist (bin M/2). The input is real, so the other M/2−1 bins of an M-point FFT are mirror
images and are not computed.

- **Prototype filter.** T·M taps of a low-pass filter (`pfb::KaiserTaps`, cutoff fs/M), stored as
  2T rows of M/2.
- **Frames.** The input is read as rows of M/2 samples. Output row r comes from a frame of M
  samples: its first half is the sum over the even tap rows k of `taps[k] * input[r + k]`
  (element by element), its second half the same over the odd tap rows.
- **FFT.** Each frame goes through an M-point real FFT, giving the M/2+1 bins of output row r.
- **2x oversampling.** Successive rows start M/2 samples apart, so each bin's output runs at
  2·fs/M, twice the channel spacing, and covers −1 to +1 channel spacings around the bin
  center. Each bin is flat to ±0.9 channel spacings, so neighboring bins overlap and every
  frequency lies in the flat part of some bin (see [Frequency response](#frequency-response)).
- **Half swap.** On odd rows the two halves of the frame are swapped before the FFT. Starting a
  frame M/2 samples later would otherwise flip the sign of every odd bin on alternate rows;
  the swap keeps each bin's rows a continuous baseband signal.

Two implementations compute the same output:

- **fused** (M ≤ 64): one kernel does the FIR and the FFT. The FFT is a cuFFTDx thread FFT,
  which keeps a whole M-point transform in one thread's registers; that only fits for small M.
  The frames never leave the chip.
- **separate** (any even M): a FIR kernel writes the frames to GPU memory and cuFFT transforms
  them, so the frames make one extra round trip through GPU memory. At M=64 the fused kernel is
  3.2× faster.

### Glossary

| Term | Meaning |
|---|---|
| M | number of channels of the filter bank (FFT length); M/2+1 bins come out |
| T | taps per branch: the prototype filter has T·M taps |
| Q | output rows per batch (a multiple of 8 for the fused path, 10 for the separate one) |
| row | one output time step: M/2+1 bins for each ADC channel, every M/2 input samples |
| bin | one frequency channel; bin k is centered at k·fs/M |
| channel spacing | fs/M, the distance between bin centers |
| frame | the M filtered samples that one row's FFT transforms |
| ADC channel | one of the 4 signals interleaved in a digitizer card's samples |
| digitizer card | one input stream of 4 interleaved ADC channels; one `Channelizer` each |
| offset binary | unsigned samples where 32768 means zero |
| `input_samples()` | input samples one batch reads: (Q − 1 + 2T)·M/2 |
| `input_advance()` | input samples between the starts of successive batches: Q·M/2 |
| Gsps | 10⁹ samples per second, summed over all 12 ADC channels of the HAARP system |
| % of nominal | bandwidth used as a share of the hardware's nominal maximum: GPU memory (DRAM) for the kernels, PCIe for end to end |

### Reading the code

1. `common/pfb_reference.hpp`: the whole filter bank as one short function in plain fp64 C++.
2. `include/pfb/pfb.cuh`: the low-level API, the same formulas in its header comment, and
   `FusedKernel`.
3. `include/pfb/fir_kernel.cuh`: the separate path's FIR kernel, then `pfb::Separate` at the end
   of `pfb.cuh`.
4. `include/pfb/channelizer.hpp` and `src/channelizer.cu`: buffers, CUDA stream and host copies.
5. `tests/test_pfb.cpp`: what is checked, and how.

## Layout

```
include/pfb/channelizer.hpp  pfb::Channelizer: one digitizer card's PFB with host copies (operational API)
include/pfb/pfb.cuh          low-level API: pfb::Fused, pfb::Separate, pfb::SeparatePlan, size helpers
include/pfb/fir_kernel.cuh   FIR kernel of the separate path
include/pfb/sizes.hpp        sizes, limits, UsesFused(M): plain C++, shared by host and GPU code
include/pfb/kaiser.hpp       pfb::KaiserTaps: Kaiser-windowed prototype filter (unit DC gain)
src/channelizer.cu           Channelizer implementation (library libpfb)
tests/test_pfb.cpp           correctness test; tests/data/golden_M<M>_T<T>.bin (800/32, 64/32)
tools/pfb_response.cpp       frequency response measurement, written to CSV (not a test)
bench/                       benchmarkpfb (kernels only), benchmarkpfbio (streaming), run_all.sh
common/                      fp64 reference model, benchmark helpers (bench.hpp)
```

## Build

Needs the CUDA toolkit, CMake ≥ 3.18 and a host compiler that nvcc supports. Tested with CUDA
13.0 (gcc 12.3) and CUDA 11.5 (gcc 10.5), with CMake 3.22 and 4.4. cuFFTDx is downloaded at
configure time (pinned nvidia-mathdx wheel); to build offline, pass
`-DFETCHCONTENT_SOURCE_DIR_MATHDX=<unpacked wheel>`.

```
cmake -S . -B build                  # default M=800, T=32
cmake --build build -j
cd build && ctest
```

To pick a toolkit other than the first `nvcc` on your `PATH`, add
`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc` (for example). CUDA 11.5 needs gcc ≤ 10,
since gcc 11's `<functional>` does not compile with it: add `-DCMAKE_CUDA_HOST_COMPILER=g++-10`.

| Option | Default | |
|---|---|---|
| `PFB_M` | 800 | channels M (even); M ≤ 64 builds the fused kernel |
| `PFB_T` | 32 | taps per branch T |
| `CMAKE_CUDA_ARCHITECTURES` | this machine's GPU | exactly one architecture (cuFFTDx is tuned for it) |
| `PFB_BUILD_TESTS` | ON | build the test, benchmarks and `pfb_response` (OFF when added with `add_subdirectory`) |

By default CMake asks `nvidia-smi` for the GPU's compute capability and builds for it (86 for the
RTX A2000). To build for another GPU, or on a machine without one or with two different kinds, set
`CMAKE_CUDA_ARCHITECTURES` to the compute capability without the dot, e.g. 89 for an RTX 4090 or
90 for an H100. It must be a single number (not `native` or `all`). The tests and benchmarks
have run on 86 only; with CUDA 13.0 the code also compiles, without warnings, for 75, 80, 89, 90,
100 and 120.

`Channelizer` runs at the build's M and T. The low-level `pfb::Fused<M, T>` / `pfb::Separate<M, T>`
templates work for any supported M and T.

## Usage

```cpp
#include "pfb/channelizer.hpp"
#include "pfb/kaiser.hpp"

auto taps = pfb::KaiserTaps(T * M, pfb::DEFAULT_ATTEN_DB, 1. / M);  // 120 dB stopband
pfb::Channelizer c(M, T, Q, taps);                                   // Q output rows per batch
std::vector<uint16_t> in(4 * c.input_samples());                     // interleaved ADC samples
std::vector<std::complex<float>> out(4 * c.nbins() * Q);
pfb::PfbOutput view{out.data(), (size_t)Q, (size_t)c.nbins() * Q};   // [adc][bin][row]
c.process(in.data(), view);              // or launch() ... wait() to overlap several digitizer cards
// next batch starts c.input_advance() samples later
```

`KaiserTaps` has unit DC gain. HAARP's `pfb` service scales the taps by 1/16384 for its int16
input. The golden test uses 2/16384, the scale its golden files were recorded with.

Output element (adc, bin, row) goes to `data[adc * adc_stride + bin * bin_stride + row]`.
`set_bins()` selects which bins are copied back; the other slots are left untouched. Host buffers
should be pinned for the copies to be asynchronous. Errors throw `std::invalid_argument` or
`std::runtime_error`.

### In another CMake project

```cmake
set(PFB_M 800 CACHE STRING "")       # the filter bank size is fixed at build time
set(PFB_T 32 CACHE STRING "")
add_subdirectory(pfb-gpu)            # builds only the library
target_link_libraries(your_app PRIVATE pfb)
```

`channelizer.hpp` is plain C++, so `your_app` can be compiled by the host compiler alone; it gets
`PFB_M` and `PFB_T` as macros, and the M and T passed to `Channelizer` must equal them. Code that
uses the low-level `pfb.cuh` directly must be CUDA compiled as C++17. Build in Release
(`-DCMAKE_BUILD_TYPE=Release`) for the speeds above.

## Correctness

`test_pfb` (run by `ctest`) checks the build's M and T through `Channelizer`:

1. all bins against an fp64 reference model (`common/pfb_reference.hpp`);
2. every third bin into a strided output with padded rows: selected bins match the model and
   nothing else in the buffer is written;
3. bad arguments throw `std::invalid_argument`, including a Q too large for the separate path's
   32-bit indexing (4·Q·M must stay below 2³¹, about 3.3 s of signal at 80 Msps);
4. the output matches the recorded `tests/data/golden_M<M>_T<T>.bin` (the default 120 dB Kaiser taps),
   where one exists for the build's M and T (800/32 and 64/32); `test_pfb --generate` records one.

The benchmarks also check their outputs against the model: `benchmarkpfb` both paths at every M,
`benchmarkpfbio` what arrives in host memory.

## Frequency response

`pfb_response [output directory]` (created if missing) measures the response at the build's M and T through
`Channelizer`, with real int16 tones and the default taps (`pfb::DEFAULT_ATTEN_DB`, 120 dB), and writes two CSV files:

- `response_M<M>_T<T>_detail.csv`: about 500 tones across one channel around bin M/8, with the
  bins within 4 of it, so the response shape from -4.5 to 4.5 channels at 0.002-channel steps;
- `response_M<M>_T<T>_band.csv`: a tone every 1/8 channel from DC to Nyquist, with the bins
  within 3 of each tone.

Frequencies are in channel spacings (fs/M): bin k is centered at k, and its output covers -1 to 1
(sample rate 2 fs/M). Columns: `tone`, `bin`, `offset` (tone - bin), `out_freq` (where the tone
lands in the bin's output; offsets beyond ±1 fold back), `measured_db`, and `predicted_db`
(|H(offset)| computed from the taps in fp64). Measuring each bin's output at the tone's exact
output frequency, averaged over 1000 rows, puts the int16 noise floor at about -135 dB (M=64)
and -146 dB (M=800).

To sweep one bin in Hz, like haarp's `siggen_tone_power.py` does with a signal generator and the
real ADCs, but with synthetic tones:

```
pfb_response --bin 100 --start 9.7e6 --stop 10.3e6 --step 25e3 --csv sb100.csv
```

It prints a table and, with `--csv`, also writes it to a file. Columns: `freq_hz`,
`offset_from_center_hz`, `aliased_offset_hz` (where the tone lands in the bin's output),
`a_rms`, `power_db` (the tone's amplitude in the output, as `siggen_tone_power.py` reports it),
`gain_db` (measured |H|) and `predicted_db`. Options: `--fs` (ADC sample rate, default 80e6),
`--amp` (tone amplitude in counts, default 32000), `--rows` (rows averaged per tone, default 1000),
and `--scale` (tap gain, default 1; haarp's `pfb` service uses 1/16384, so set it to compare
`power_db` with that service's output). Tones are rounded to a grid of fs/(M·rows), and a tone on
a bin center is moved by one grid step; the frequency actually used is the one reported.

Measured 2026-11-11, M=64 and M=800 with T=32 and the default 120 dB taps (the response in
channel spacings is the same for both):

| Offset from bin center | Response |
|---|---|
| 0 to 0.9 | flat to 0.005 dB |
| 0.981 | -3 dB |
| 1.0 (output Nyquist) | -6 dB; the channels overlap by design |
| 1.1 | -64.5 dB |
| 1.119 | -100 dB |
| 1.123 | -120 dB |
| beyond 1.15 | -122 dB at most |

Measured and predicted agree to 0.01 dB wherever the response is above -80 dB, including the DC
and Nyquist bins. In the stopband, M=800 tracks the prediction to within about 0.3 dB down to
-130 dB; at M=64 the noise floor is only about 13 dB below the stopband, so the measurement
there reads up to 2 dB high.

## Benchmarks

Throughput is given in Gsps for the HAARP system's 12 ADC channels (3 digitizer cards × 4). Each
benchmark also reports how much of the hardware's nominal capacity it uses, for the resource that
bounds it:

- **Kernels only:** the minimum GPU memory (DRAM) traffic each path must move, per second, as a
  share of the GPU's nominal DRAM bandwidth (from its memory clock and bus width; 288 GB/s on
  the RTX A2000). The separate path also writes its frames to DRAM and reads them back.
- **End to end:** PCIe traffic in each direction as a share of the link's nominal bandwidth per
  direction (from its maximum generation and width, read through NVML; PCIe 4.0 x16 is
  31.5 GB/s).

- `benchmarkpfb`: kernels only (data already on the GPU), T=32, 100 ms batches, M from 8 to 1024.
- `benchmarkpfbio`: streaming, including PCIe copies, T=32, 100 ms batches, the same M sweep.
  Each digitizer card's batch goes through the steps of `Channelizer` (copy in, filter bank,
  pitched copy of the selected bins back to pinned host memory) on its own CUDA stream, double
  buffered with batches back to back: the sustained rate of a continuous stream. All bins, or a
  quarter of them, copied back.

### Reproducing

```
bench/run_all.sh                       # writes results/<gpu>_<date>/
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" bench/run_all.sh                 # another GPU
CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc" bench/run_all.sh   # a toolkit
CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=g++-10" bench/run_all.sh             # CUDA 11.5
```

The script builds M=800 and M=64 (both T=32) in `build-bench/` and runs `test_pfb` for each,
and `benchmarkpfb` and `benchmarkpfbio` once. It saves every program's raw output, plus
`system.txt`, which records:

- the GPU, driver, power limit, maximum clocks and PCIe link;
- the CUDA, compiler and CMake versions, and the git commit;
- any other processes using the GPU, which slow the timings down. The script also warns about
  these.

It stops if any correctness check fails. Use `CUDA_VISIBLE_DEVICES` to pick a GPU other than 0.

### Results

RTX A2000 12GB (70 W power limit), CUDA 13.0 with gcc 12.3, driver 580.178.04, PCIe 4.0 x16, no
other processes on the GPU; measured 2026-11-25 with `bench/run_all.sh`. The raw output and
machine details are in
[results/NVIDIA_RTX_A2000_12GB_2026-11-25_cuda13.0](results/NVIDIA_RTX_A2000_12GB_2026-11-25_cuda13.0).

Throughput in Gsps for all 12 ADC channels (real time needs 0.96), T=32, 100 ms batches, each
figure the mean ± standard deviation over 9 trials:

- **Kernel throughput** (`benchmarkpfb`): the filter bank alone, data already on the GPU, using
  the path the library picks (fused for M ≤ 64, separate above). Each trial is the mean of 20
  batches.
- **Streaming** (`benchmarkpfbio`): including the PCIe copies in both directions, double
  buffered with batches back to back. Each trial is 20 batches (2 s of signal); all M/2+1 bins
  copied back, or a quarter of them, (M/2+1)/4 rounded down.

| M | Kernel throughput | Streaming, all bins | Streaming, a quarter of the bins |
|---:|---:|---:|---:|
| 8 | 4.69 ± 0.02 (fused) | 2.51 ± 0.00 | 4.45 ± 0.02 (1 bin) |
| 16 | 8.23 ± 0.01 (fused) | 2.76 ± 0.02 | 7.47 ± 0.02 (2 bins) |
| 32 | 13.82 ± 0.11 (fused) | 2.95 ± 0.00 | 10.39 ± 0.09 (4 bins) |
| 50 | 15.87 ± 0.22 (fused) | 2.98 ± 0.00 | 10.46 ± 0.10 (6 bins) |
| 64 | 19.84 ± 0.24 (fused) | 3.00 ± 0.00 | 10.39 ± 0.02 (8 bins) |
| 80 | 6.66 ± 0.12 (separate) | 2.87 ± 0.03 | 5.82 ± 0.19 (10 bins) |
| 128 | 6.27 ± 0.12 (separate) | 2.91 ± 0.00 | 5.70 ± 0.02 (16 bins) |
| 256 | 7.42 ± 0.25 (separate) | 2.95 ± 0.01 | 6.56 ± 0.11 (32 bins) |
| 400 | 5.22 ± 0.06 (separate) | 2.91 ± 0.01 | 4.92 ± 0.05 (50 bins) |
| 512 | 7.26 ± 0.15 (separate) | 2.97 ± 0.01 | 6.61 ± 0.08 (64 bins) |
| 800 | 4.77 ± 0.10 (separate) | 2.90 ± 0.05 | 4.46 ± 0.02 (100 bins) |
| 1024 | 7.03 ± 0.19 (separate) | 2.95 ± 0.02 | 6.33 ± 0.02 (128 bins) |

With all bins copied back, streaming levels off near 3 Gsps at every M: the output is about four
times the input, and the device-to-host copies run at 74–80% of the nominal 31.5 GB/s.
With a quarter of the bins the two directions carry the same traffic, and the slower of the
kernels and the PCIe link sets the rate: the fused kernels at M = 32–64 are fast enough for PCIe
to bind (about 66% of nominal each way), while at small M and on the separate path the
kernels do. The 9 streaming trials (about 0.6 s each) run back to back, which takes this board to
its 70 W power limit and lowers its clocks; the kernel trials are short bursts, so where the
kernels bind, streaming lands somewhat below the kernel figure.

The ± figures are the spread within one run. Between runs the spread is larger: an earlier run the
same day, started on a cooler board, gave kernel figures 3–5% higher (boost clocks fall as the
board warms), while the PCIe-bound streaming figures agreed within 1%.

Kernels only, both paths (`benchmarkpfb`), with each path's minimum DRAM traffic at the mean
throughput and its share of the nominal 288 GB/s:

| M | fused Gsps | DRAM GB/s | % of nominal | separate Gsps | DRAM GB/s | % of nominal |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4.69 ± 0.02 | 56 | 20 | 2.50 ± 0.04 | 70 | 24 |
| 16 | 8.23 ± 0.01 | 90 | 31 | 4.04 ± 0.01 | 109 | 38 |
| 32 | 13.82 ± 0.11 | 145 | 50 | 5.36 ± 0.28 | 142 | 49 |
| 50 | 15.87 ± 0.22 | 164 | 57 | 7.07 ± 0.27 | 186 | 65 |
| 64 | 19.84 ± 0.24 | 203 | 71 | 6.37 ± 0.33 | 167 | 58 |
| 80 |  | |  | 6.66 ± 0.12 | 175 | 61 |
| 128 |  | |  | 6.27 ± 0.12 | 164 | 57 |
| 256 |  | |  | 7.42 ± 0.25 | 193 | 67 |
| 400 |  | |  | 5.22 ± 0.06 | 136 | 47 |
| 512 |  | |  | 7.26 ± 0.15 | 189 | 66 |
| 800 |  | |  | 4.77 ± 0.10 | 124 | 43 |
| 1024 |  | |  | 7.03 ± 0.19 | 183 | 64 |

At M=64 the fused kernel is 3.1× faster than the separate path: it moves less data (no frames
through DRAM), and moves it at 71% of the nominal DRAM bandwidth against 58% for the separate path.

## License

The code in this repository is MIT licensed; see [LICENSE](LICENSE).

It depends on NVIDIA software under NVIDIA's own licenses, which this repository does not include
or redistribute:

- **cuFFTDx** (the nvidia-mathdx package) is downloaded from PyPI when CMake configures the build.
  It is covered by the *License Agreement for NVIDIA Math Libraries Software Development Kits*,
  included in the download as `nvidia_mathdx-25.6.0.dist-info/licenses/LICENSE.txt` (in
  `build/_deps/mathdx-src/`). Read it before you distribute anything built with it.
- **CUDA runtime and cuFFT** come with the CUDA toolkit you install, under its license.

## Acknowledgements

This project was supported in part by the Subauroral Geophysical Observatory award (#2054361) from
the U.S. National Science Foundation. The Subauroral Geophysical Observatory (SAGO) is operated by
the University of Alaska Fairbanks.

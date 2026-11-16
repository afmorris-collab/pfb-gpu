#!/usr/bin/env bash
# Reproduce the README's results: build the two published configurations, run the correctness
# test and both benchmarks, and save the raw output with a description of the machine.
#
# Usage:  bench/run_all.sh [output directory]
#         (default: results/<gpu>_<date>)
# Extra CMake arguments go in CMAKE_ARGS, e.g. for another GPU or an older CUDA toolkit:
#         CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" bench/run_all.sh
#         CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc" bench/run_all.sh
#         CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=g++-10" bench/run_all.sh     (CUDA 11.5)
# The benchmarks use CUDA device 0; pick another with CUDA_VISIBLE_DEVICES.
#
# Builds:  M=800, T=32 (separate path, as HAARP runs it) and M=64, T=32 (fused path).
# Output:  system.txt          GPU, driver, clocks, power limit, PCIe link, toolchain, git commit,
#                              and any other processes using the GPU
#          test_pfb_M<M>.txt   correctness test for each build
#          benchmarkpfb.txt    kernels only, M from 8 to 1024 (does not depend on the build's M)
#          benchmarkpfbio_M<M>.txt  end to end through Channelizer for each build
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # CUDA device numbers match nvidia-smi's
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
CONFIGS=("800 32" "64 32")                   # M T

gpu_name=$(nvidia-smi -i "$GPU_ID" --query-gpu=name --format=csv,noheader | tr ' ' '_' | tr -cd '[:alnum:]_-')
OUT="${1:-results/${gpu_name}_$(date +%Y-%m-%d)}"
mkdir -p "$OUT"
echo "Writing results to $OUT"

# ---- Build ----------------------------------------------------------------------------------
for cfg in "${CONFIGS[@]}"; do
    read -r M T <<< "$cfg"
    dir="build-bench/M${M}_T${T}"
    # shellcheck disable=SC2086  # CMAKE_ARGS is a list of arguments
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Release -DPFB_M="$M" -DPFB_T="$T" ${CMAKE_ARGS:-} > /dev/null
    if ! cmake --build "$dir" -j > /dev/null; then
        echo "Build failed in $dir. With CUDA 11.5, use gcc <= 10:" >&2
        echo "  CMAKE_ARGS=\"-DCMAKE_CUDA_HOST_COMPILER=g++-10\" $0" >&2
        exit 1
    fi
    echo "Built M=$M T=$T in $dir"
done

# ---- Describe the machine -------------------------------------------------------------------
first_build="build-bench/M800_T32"
{
    echo "date:        $(date -Iseconds)"
    if git rev-parse --git-dir > /dev/null 2>&1; then
        echo "commit:      $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- . ':!results' || echo ' (uncommitted changes)')"
    else
        echo "commit:      unknown (not a git checkout)"
    fi
    echo "host:        $(uname -srm)"
    echo "cpu:         $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    nvcc_path=$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "$first_build/CMakeCache.txt")
    echo "nvcc:        $nvcc_path, $("$nvcc_path" --version | grep -o 'release [0-9.]*, V[0-9.]*')"
    host_cxx=$(sed -n 's/^CMAKE_CUDA_HOST_COMPILER:[A-Z]*=//p' "$first_build/CMakeCache.txt")
    [ -n "$host_cxx" ] || host_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$first_build/CMakeCache.txt")
    echo "host cxx:    $("$host_cxx" --version | head -1)"
    echo "cmake:       $(cmake --version | head -1)"
    echo "cmake args:  ${CMAKE_ARGS:-(none)}"
    echo
    echo "GPU $GPU_ID:"
    nvidia-smi -i "$GPU_ID" --format=csv --query-gpu=name,driver_version,memory.total,power.limit,power.default_limit,clocks.max.sm,clocks.max.memory,pcie.link.gen.max,pcie.link.width.max,pcie.link.width.current,persistence_mode
    echo "(the current PCIe generation drops when the GPU is idle, so the maximum is given)"
    echo
    echo "Other processes using the GPU at the start (they make the timings slower):"
    nvidia-smi -i "$GPU_ID" --format=csv --query-compute-apps=pid,process_name,used_memory
} > "$OUT/system.txt"

others=$(nvidia-smi -i "$GPU_ID" --format=csv,noheader --query-compute-apps=pid | wc -l)
if [ "$others" -gt 0 ]; then
    echo "WARNING: $others other process(es) are using the GPU; the timings will be low (see $OUT/system.txt)"
fi

# ---- Run ------------------------------------------------------------------------------------
for cfg in "${CONFIGS[@]}"; do
    read -r M T <<< "$cfg"
    echo "test_pfb M=$M T=$T"
    "build-bench/M${M}_T${T}/test_pfb" > "$OUT/test_pfb_M${M}.txt"
done

echo "benchmarkpfb (kernels only, M sweep)"
"$first_build/benchmarkpfb" > "$OUT/benchmarkpfb.txt"

for cfg in "${CONFIGS[@]}"; do
    read -r M T <<< "$cfg"
    echo "benchmarkpfbio M=$M T=$T"
    "build-bench/M${M}_T${T}/benchmarkpfbio" > "$OUT/benchmarkpfbio_M${M}.txt"
    if grep -q -E "FAIL|WRONG" "$OUT/benchmarkpfbio_M${M}.txt"; then  # it does not set an exit code
        echo "benchmarkpfbio M=$M: a check failed, see $OUT/benchmarkpfbio_M${M}.txt" >&2
        exit 1
    fi
done

echo "Done: all checks passed. Results in $OUT"

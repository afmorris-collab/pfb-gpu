cmake_minimum_required(VERSION 3.18)

# One CUDA architecture (cuFFTDx tunes the fused kernel's FFT for it). Default: the GPU in this
# machine, from nvidia-smi (compute capability 8.6 -> 86). Set CMAKE_CUDA_ARCHITECTURES (or the
# CUDAARCHS environment variable) to build for another GPU, or on a machine without one.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE pfb_caps RESULT_VARIABLE pfb_rc ERROR_QUIET)
    string(REPLACE "." "" pfb_caps "${pfb_caps}")
    string(REGEX MATCHALL "[0-9]+" pfb_caps "${pfb_caps}")
    if(pfb_caps)
        list(REMOVE_DUPLICATES pfb_caps)
    endif()
    list(LENGTH pfb_caps pfb_ncaps)
    if(NOT pfb_rc EQUAL 0 OR NOT pfb_ncaps EQUAL 1)
        message(FATAL_ERROR "Could not detect a single GPU architecture with nvidia-smi "
                            "(found '${pfb_caps}'). Set it, e.g. -DCMAKE_CUDA_ARCHITECTURES=86.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${pfb_caps} CACHE STRING "CUDA architecture (detected with nvidia-smi)")
    message(STATUS "CUDA architecture ${pfb_caps}, detected with nvidia-smi")
endif()

project(pfb_gpu LANGUAGES CXX CUDA)

# Built on its own, or added to another project with add_subdirectory (then only the library).
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(PFB_TOP_LEVEL ON)
else()
    set(PFB_TOP_LEVEL OFF)
endif()
option(PFB_BUILD_TESTS "Build the test, benchmarks and pfb_response" ${PFB_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(PFB_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
# Programs linked by nvcc (the benchmarks) get no runtime path to the toolkit's libraries, since
# CMake treats nvcc's library directory as implicit. Add it, so that a toolkit outside the system
# library path (e.g. /usr/local/cuda-13.0) works without LD_LIBRARY_PATH or ldconfig.
list(APPEND CMAKE_BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")

# cuFFTDx (header-only, via the nvidia-mathdx wheel), pinned. Used by the fused PFB kernel.
# NVIDIA's license, not MIT (see README); it is downloaded here, never stored in this repository.
# Offline: -DFETCHCONTENT_SOURCE_DIR_MATHDX=<unpacked wheel>.
include(FetchContent)
if(POLICY CMP0135)    # CMake >= 3.24: give extracted files the time of extraction
    cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
    mathdx
    URL https://files.pythonhosted.org/packages/20/1a/a418b8c1adc58abd87fd69414c19883af5c1b10514e3dbfcc27cde831b13/nvidia_mathdx-25.6.0-py3-none-any.whl
    URL_HASH SHA256=22e6ad5d0d005f836be5cbd14e836cf2e9ea42c82deb602707246ce8198eaa96
    # The wheel holds only headers. Pointing SOURCE_SUBDIR at a directory without a
    # CMakeLists.txt makes FetchContent_MakeAvailable download it without add_subdirectory.
    SOURCE_SUBDIR no-cmake
)
FetchContent_MakeAvailable(mathdx)
set(MATHDX_INCLUDE_DIR ${mathdx_SOURCE_DIR}/nvidia/mathdx/include)

# PFB size, fixed at build time: M channels, T taps per branch. M <= 64 uses the fused FIR+FFT
# kernel (pfb::FUSED_MAX_M in sizes.hpp), larger M the separate FIR and cuFFT kernels. Default: 800/32, as HAARP runs it.
set(PFB_M 800 CACHE STRING "PFB channels M")
set(PFB_T 32 CACHE STRING "PFB taps per branch T")

list(LENGTH CMAKE_CUDA_ARCHITECTURES PFB_NARCH)
if(NOT PFB_NARCH EQUAL 1)
    message(FATAL_ERROR "PFB needs exactly one CUDA architecture, got CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()
string(REGEX MATCH "^[0-9]+" PFB_ARCH "${CMAKE_CUDA_ARCHITECTURES}")
math(EXPR PFB_CUFFTDX_SM "${PFB_ARCH} * 10")

# ---- Library: Channelizer plus the header-only low-level API (include/pfb) ----
add_library(pfb STATIC src/channelizer.cu)
target_include_directories(pfb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${MATHDX_INCLUDE_DIR})
target_link_libraries(pfb PUBLIC CUDA::cudart CUDA::cufft)
target_compile_features(pfb PUBLIC cxx_std_17)    # CUDA code using pfb.cuh also needs C++17
# CUFFTDX_DISABLE_CUTLASS_DEPENDENCY: skip commondx's CUTLASS interop; we don't use CUTLASS types.
target_compile_definitions(pfb PUBLIC
    CUFFTDX_DISABLE_CUTLASS_DEPENDENCY PFB_M=${PFB_M} PFB_T=${PFB_T} PFB_CUFFTDX_SM=${PFB_CUFFTDX_SM})

if(NOT PFB_BUILD_TESTS)
    return()
endif()

# ---- Correctness test: the build's M and T (fp64 model; golden output where recorded) ----
enable_testing()
add_executable(test_pfb tests/test_pfb.cpp)
target_include_directories(test_pfb PRIVATE common)
target_link_libraries(test_pfb PRIVATE pfb)
target_compile_definitions(test_pfb PRIVATE PFB_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
add_test(NAME test_pfb COMMAND test_pfb)

# pfb_response: frequency response at the build's M and T, written to CSV (not a pass/fail test)
add_executable(pfb_response tools/pfb_response.cpp)
target_link_libraries(pfb_response PRIVATE pfb)

# ---- Benchmarks ----
# benchmarkpfb: kernels only (no PCIe), T=32, M sweep, fused vs separate
# benchmarkpfbio: end to end incl. PCIe through Channelizer, at the build's M and T
foreach(bench benchmarkpfb benchmarkpfbio)
    add_executable(${bench} bench/${bench}.cu)
    target_include_directories(${bench} PRIVATE common)
    target_link_libraries(${bench} PRIVATE pfb)
endforeach()
target_link_libraries(benchmarkpfbio PRIVATE CUDA::nvml)    # PCIe link generation and width

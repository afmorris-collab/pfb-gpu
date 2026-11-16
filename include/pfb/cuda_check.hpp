// Error checks for the PFB library: CUDA and cuFFT failures throw std::runtime_error.
#pragma once

#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cufft.h>

namespace pfb::detail {

// cuFFT has no error-string function; name the common codes, number the rest.
inline std::string CufftErrorName(cufftResult r)
{
    switch (r) {
    case CUFFT_INVALID_PLAN:    return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:    return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:    return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:   return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:  return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:     return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:    return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:    return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:  return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE:  return "CUFFT_INVALID_DEVICE";
    case CUFFT_NO_WORKSPACE:    return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:   return "CUFFT_NOT_SUPPORTED";
    default:                    return "cufftResult " + std::to_string(r);
    }
}

}  // namespace pfb::detail

#define PFB_CUDA_CHECK(x) do { cudaError_t pfb_e_ = (x); if (pfb_e_ != cudaSuccess) \
    throw std::runtime_error(std::string("CUDA error ") + cudaGetErrorString(pfb_e_) + " at " + \
                             __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

#define PFB_CUFFT_CHECK(x) do { cufftResult pfb_r_ = (x); if (pfb_r_ != CUFFT_SUCCESS) \
    throw std::runtime_error("cuFFT error " + pfb::detail::CufftErrorName(pfb_r_) + " at " + \
                             __FILE__ + ":" + std::to_string(__LINE__)); } while (0)

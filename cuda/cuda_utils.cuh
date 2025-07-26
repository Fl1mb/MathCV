#ifndef CUDA_UTILS_H_
#define CUDA_UTILS_H_


#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
}

class CUDAChecker {
public:
    static bool isAvailable() {
        static bool checked = false;
        static bool available = false;
        static int count = 0;
        
        if (!checked) {
            cudaError_t err = cudaGetDeviceCount(&count);
            available = (err == cudaSuccess && count > 0);
            checked = true;
        }
        return available;
    }

};



#else
class CUDAChecker {
public:
    static bool isAvailable() { return false; }
};

#endif


#endif //CUDA_UTILS_H_
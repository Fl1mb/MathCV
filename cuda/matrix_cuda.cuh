#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include <vector>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA_ERRORS(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

namespace MathCV {
    // Объявления функций работы с памятью
    template<typename T> T* deviceAllocMatrix(size_t rows, size_t cols);
    template<typename T> void copyMatrixToDevice(const T* h_matrix, T* d_matrix, size_t rows, size_t cols);
    template<typename T> void copyMatrixToHost(const T* d_matrix, T* h_matrix, size_t rows, size_t cols);
    template<typename T> void freeDeviceMatrix(T* matrix);

    // Объявления ядер
    template <typename T> __global__ void matrixMulKernel(const T* A, const T* B, T* C, size_t a_rows, size_t a_cols, size_t b_cols);
    template<typename T> __global__ void matrixAddKernel(const T* A, const T* B, T* C, size_t rows, size_t cols);
    template<typename T> __global__ void matrixSubKernel(const T* A, const T* B, T* C, size_t rows, size_t cols);
    template<typename T> __global__ void matrixTransposeKernel(const T* A, T* C, size_t rows, size_t cols);
    template<typename T> __global__ void luDecompositionKernel(T* A, T* L, T* U, size_t n);
    template<typename T> __global__ void determinantKernel(const T* U, T* det, size_t n);
    template<typename T> __global__ void matrixOfMinorsKernel(const T* A, T* minors, size_t n);
    template<typename T> __global__ void scalarMultiplyKernel(T* A, double scalar, size_t rows, size_t cols);

    // Объявления оберток для вызова ядер
    template<typename T> void MatrixMul(const T* A, const T* B, T* C, size_t a_rows, size_t a_cols, size_t b_cols);
    template<typename T> void matrixAdd(const T* A, const T* B, T* C, size_t rows, size_t cols);
    template<typename T> void matrixSub(const T* A, const T* B, T* C, size_t rows, size_t cols);
    template<typename T> void matrixTranspose(const T* A, T* C, size_t rows, size_t cols);
    template<typename T> void inverseMatrixGPU(const T* d_A, T* d_invA, size_t n);
    template<typename T> T computeDet(const T* h_A, size_t n);

    template<typename T> __device__ T computeSubmatrixDet(const T* mat, size_t size, T* buffer);
    
}

#endif // MATRIX_CUDA_H
#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

//Macros for CUDA assert
#define CHECK_CUDA_ERRORS(call){\
    cudaError_t err = call; \ 
    if(err !=  cudaSuccess){ \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

namespace MathCV{
    // Tools for working with memory

    //Allocate Memory on GPU for matrix rows x cols
    template<typename T>
    T* deviceAllocMatrix(size_t rows, size_t cols){
        T* d_matrix;
        CHECK_CUDA_ERRORS(cudaMalloc(&d_matrix, rows * cols * sizeof(T)));
        return d_matrix;
    }

    //Copy matrix from CPU to GPU
    template<typename T>
    void copyMatrixToDevice(const T* h_matrix, T* d_matrix, size_t rows, size_t cols){
        CHECK_CUDA_ERRORS(cudaMemcpy(
            d_matrix,
            h_matrix,
            rows * cols * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }

    //Copy matrix from GPU to CPU
    template<typename T>
    void copyMatrixToHost(const T* h_matrix, T* d_matrix, size_t rows, size_t cols){
        CHECK_CUDA_ERRORS(cudaMemcpy(
            d_matrix,
            h_matrix,
            rows * cols * sizeof(T),
            cudaMemcpyDeviceToHost
        ));
    }

    //free memory
    template<typename T>
    void freeDeviceMatrix(T* matrix){
        CHECK_CUDA_ERRORS(cudaFree(matrix));
    }

    // =============================================
    // Ядра (Kernels) для матричных операций
    // =============================================


    //Умножение матриц
    template <typename T>
    __global__ void matrixMulKernel(
        const T* A, const T* B, T* C, 
        size_t a_rows, size_t a_cols, size_t b_cols
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < a_rows && col < b_cols) {
            T sum = 0;
            for (int k = 0; k < a_cols; ++k) {
                sum += A[row * a_cols + k] * B[k * b_cols + col];
            }
            C[row * b_cols + col] = sum;
        }
    }

    //Сложение матриц
    template<typename T>
    __global__ void matrixAddKernel(
        const T* A, const T* B, T* C,
        size_t rows, size_t cols
    ){
        int idx = blockIdx.x + blockDim.x + threadIdx.x;
        if(idx < rows * cols){
            C[idx] = A[idx] + B[idx];
        }
    }

    //Транспонирование матрицы
    template<typename T>
    __global__ void matrixTransposeKernel(
        const T* A, T* C,
        size_t rows, size_t cols
    ){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows && col < cols) {
            C[col * rows + row] = A[row * cols + col];
        }
    }
    // =============================================
    // Обертки для вызова ядер
    // =============================================

    //Умножение матриц: C = A * B
    template<typename T>
    void MatrixMul(
        const T* A, const T* B, T* C,
        size_t a_rows, size_t a_cols, size_t b_cols
    ){
        // Выделяем память на GPU
        T *d_A = deviceAllocMatrix<T>(a_rows, a_cols);
        T *d_B = deviceAllocMatrix<T>(a_cols, b_cols);
        T *d_C = deviceAllocMatrix<T>(a_rows, b_cols);

        // Копируем данные на GPU
        copyMatrixToDevice(A, d_A, a_rows, a_cols);
        copyMatrixToDevice(B, d_B, a_cols, b_cols);

        dim3 blockSize(16, 16);
        dim3 gridSize(
            (b_cols + blockSize.x - 1) / blockSize.x,
            (a_rows + blockSize.y - 1) / blockSize.y
        );
        matrixMulKernel<T><<<gridSize, blockSize>>>(d_A, d_B, d_C, a_rows, a_cols, b_cols);

        copyMatrixToHost(d_C, C, a_rows, b_cols);

        freeDeviceMatrix(d_A);
        freeDeviceMatrix(d_B);
        freeDeviceMatrix(d_C);
    }
 
    //Сложение матриц: C = A + B
    template<typename T>
    void matrixAdd(
        const T* A, const T* B, T* C,
        size_t rows, size_t cols
    ){
        T* d_A = deviceAllocMatrix(rows, cols);
        T* d_B = deviceAllocMatrix(rows,cols);
        T* d_C = deviceAllocMatrix(rows,cols);

        copyMatrixToDevice(A, d_A, rows, cols);
        copyMatrixToDevice(B, d_B, rows, cols);


        // 1D блоки и гриды
        int blockSize = 256;
        int gridSize = (rows * cols + blockSize - 1) / blockSize;

        matrixAddKernel<T><<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

        copyMatrixToHost(d_C, C, rows, cols);

        freeDeviceMatrix(d_A);
        freeDeviceMatrix(d_B);
        freeDeviceMatrix(d_C);
    }

    //Транспонирование матрицы: C = A_T
    template<typename T>
    void matrixTranspose(
        const T* A, T* C,
        size_t rows, size_t cols
    ){
        T* d_A = deviceAllocMatrix(rows, cols);
        T* d_C = deviceAllocMatrix(cols, rows);

        copyMatrixToDevice(A, d_A, rows, cols);

        dim3 blockSize(16, 16);
        dim3 gridSize(
            (cols + blockSize.x - 1) / blockSize.x,
            (rows + blockSize.y - 1) / blockSize.y
        );

        matrixTransposeKernel<T><<<gridSize, blockSize>>>(d_A, d_C, rows, cols);
        copyMatrixToHost(d_C, C, cols, rows);

        freeDeviceMatrix(d_A);
        freeDeviceMatrix(d_C);
    }
}



#endif //MATRIX_CUDA_H
#ifndef MATRIX_OPERATIONS_H_
#define MATRIX_OPERATIONS_H_

#include "matrix_cuda.cuh"
#include "cuda_utils.cuh"

namespace MathCV
{
    template<typename T> 
    class MatrixOperations{
    public:
        static void multiply(
            const T* A, const T* B, T* C,
            size_t a_rows, size_t a_cols,
            size_t b_cols
        );

        static void summarize(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void subtraction(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void transpose(
            const T* A, T* A_T,
            size_t rows, size_t cols
        );

        static void determinant(
            const T* A, double* det,
            size_t n
        );

        static void inverse(
            const T* A, T* inv_A,
            size_t n
        );

    private:
        static void multiplyCUDA(
            const T* A, const T* B, T* C,
            size_t a_rows, size_t a_cols,
            size_t b_cols
        );

        static void multiplyCPU(
            const T* A, const T* B, T* C,
            size_t a_rows, size_t a_cols,
            size_t b_cols
        );

        static void summarizeCUDA(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void summarizeCPU(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void subtractionCUDA(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void subtractionCPU(
            const T* A, const T* B, T* C,
            size_t rows, size_t cols
        );

        static void transposeCUDA(
            const T* A, T* A_T,
            size_t rows, size_t cols
        );

        static void transposeCPU(
            const T* A, T* A_T,
            size_t rows, size_t cols
        );

        static void determinantCUDA(
            const T* A, double* det,
            size_t n
        );

        static void determinantCPU(
            const T* A, double* det,
            size_t n
        );

        static void inverseCUDA(
            const T* A, T* inv_A,
            size_t n
        );

        static void inverseCPU(
            const T* A, T* inv_A,
            size_t n
        );
    };
} // namespace MathCV



#endif //MATRIX_OPERATIONS_H_
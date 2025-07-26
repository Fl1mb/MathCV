#include "matrix_cuda.cuh"

using namespace MathCV;


template<typename T>
T* MathCV::deviceAllocMatrix(size_t rows, size_t cols){
    T* d_matrix;
    CHECK_CUDA_ERRORS(cudaMalloc(&d_matrix, rows * cols * sizeof(T)));
    return d_matrix;
}

//Copy matrix from CPU to GPU
template<typename T>
void MathCV::copyMatrixToDevice(const T* h_matrix, T* d_matrix, size_t rows, size_t cols){
    CHECK_CUDA_ERRORS(cudaMemcpy(
        d_matrix,
        h_matrix,
        rows * cols * sizeof(T),
        cudaMemcpyHostToDevice
    ));
}

//Copy matrix from GPU to CPU
template<typename T>
void MathCV::copyMatrixToHost(const T* h_matrix, T* d_matrix, size_t rows, size_t cols){
    CHECK_CUDA_ERRORS(cudaMemcpy(
        d_matrix,
        h_matrix,
        rows * cols * sizeof(T),
        cudaMemcpyDeviceToHost
    ));
}

//free memory
template<typename T>
void MathCV::freeDeviceMatrix(T* matrix){
    CHECK_CUDA_ERRORS(cudaFree(matrix));
}

// =============================================
// Ядра (Kernels) для матричных операций
// =============================================


//Умножение матриц
template <typename T>
__global__ void MathCV::matrixMulKernel(
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
__global__ void MathCV::matrixAddKernel(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows * cols){
        C[idx] = A[idx] + B[idx];
    }
}

//Умножение на скаляр
template<typename T> 
__global__ void MathCV::scalarMultiplyKernel(T* A, double scalar, size_t rows, size_t cols){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < rows && j < cols){
        A[i * cols + j] *= scalar;
    }
}

//Вычитание матриц
template<typename T>
__global__ void MathCV::matrixSubKernel(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows * cols){
        C[idx] = A[idx] - B[idx];
    }
}

//Транспонирование матрицы
template<typename T>
__global__ void MathCV::matrixTransposeKernel(
    const T* A, T* C,
    size_t rows, size_t cols
){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        C[col * rows + row] = A[row * cols + col];
    }
}


template<typename T>
__global__ void MathCV::luDecompositionKernel(
    T* A, T* L, T* U,
    size_t n
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
    if (row == col) {
        L[row * n + col] = 1.0; // Диагональ L = 1
    }
    if (row <= col) {
        float sum = 0.0;
        for (int k = 0; k < row; ++k) {
            sum += L[row * n + k] * U[k * n + col];
        }
        U[row * n + col] = A[row * n + col] - sum;
    } else {
        float sum = 0.0;
        for (int k = 0; k < col; ++k) {
            sum += L[row * n + k] * U[k * n + col];
        }
        L[row * n + col] = (A[row * n + col] - sum) / U[col * n + col];
    }
    }
}

template<typename T>
__global__ void MathCV::determinantKernel(
    const T* U, T* det, size_t n
){
    extern __shared__ float temp_det;
    if(threadIdx.x == 0){
        temp_det = 1.0;
        for(size_t i = 0; i < n; ++i){
            temp_det *= U[i * n + i];
        }
    }
    

    if(threadIdx.x == 0){
        *det = temp_det;
    }
}

template<typename T> 
__device__ T MathCV::computeSubmatrixDet(const T* mat, size_t size, T* buffer){
    if (size == 1) return mat[0];
    if (size == 2) return mat[0]*mat[3] - mat[1]*mat[2];
    
    T det = 0;
    const size_t minor_size = (size-1)*(size-1);
    
    for (size_t col = 0; col < size; ++col) {
        size_t minor_idx = 0;
        
        for (size_t i = 1; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                if (j == col) continue;
                buffer[minor_idx++] = mat[i*size + j];
            }
        }
        
        T minor_det = computeSubmatrixDet(buffer, size-1, buffer + minor_size);
        det += (col % 2 == 0 ? 1 : -1) * mat[col] * minor_det;
    }
    return det;
}

template<typename T>
__global__ void MathCV::matrixOfMinorsKernel(const T* A, T* minors, size_t n) {
    extern __shared__ __align__(sizeof(T)) unsigned char shared_mem[];
    T* buffer = reinterpret_cast<T*>(shared_mem);

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    // Каждый поток работает со своим элементом
    size_t subIdx = 0;
    for (size_t k = 0; k < n; ++k) {
        if (k == i) continue;
        for (size_t l = 0; l < n; ++l) {
            if (l == j) continue;
            buffer[threadIdx.y * blockDim.x + threadIdx.x + subIdx] = A[k * n + l];
            subIdx++;
        }
    }
    __syncthreads();

    T subDet = computeSubmatrixDet(buffer + (threadIdx.y * blockDim.x + threadIdx.x) * (n-1)*(n-1), 
                      n-1, 
                      buffer + blockDim.x * blockDim.y * (n-1)*(n-1));
    
    minors[i * n + j] = ((i + j) % 2 == 0) ? subDet : -subDet;
}

// =============================================
// Обертки для вызова ядер
// =============================================

//Умножение матриц: C = A * B
template<typename T>
void MathCV::MatrixMul(
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
    matrixMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, a_rows, a_cols, b_cols);

    copyMatrixToHost(d_C, C, a_rows, b_cols);

    freeDeviceMatrix(d_A);
    freeDeviceMatrix(d_B);
    freeDeviceMatrix(d_C);
}

//Сложение матриц: C = A + B
template<typename T>
void MathCV::matrixAdd(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    T* d_A = deviceAllocMatrix<T>(rows, cols);
    T* d_B = deviceAllocMatrix<T>(rows,cols);
    T* d_C = deviceAllocMatrix<T>(rows,cols);

    copyMatrixToDevice(A, d_A, rows, cols);
    copyMatrixToDevice(B, d_B, rows, cols);


    // 1D блоки и гриды
    int blockSize = 256;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;

    matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    copyMatrixToHost(d_C, C, rows, cols);

    freeDeviceMatrix(d_A);
    freeDeviceMatrix(d_B);
    freeDeviceMatrix(d_C);
}

//Вычитание матриц: C = A - B
template<typename T>
void MathCV::matrixSub(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    T* d_A = deviceAllocMatrix<T>(rows, cols);
    T* d_B = deviceAllocMatrix<T>(rows,cols);
    T* d_C = deviceAllocMatrix<T>(rows,cols);

    copyMatrixToDevice(A, d_A, rows, cols);
    copyMatrixToDevice(B, d_B, rows, cols);


    // 1D блоки и гриды
    int blockSize = 256;
    int gridSize = (rows * cols + blockSize - 1) / blockSize;

    matrixSubKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rows, cols);

    copyMatrixToHost(d_C, C, rows, cols);

    freeDeviceMatrix(d_A);
    freeDeviceMatrix(d_B);
    freeDeviceMatrix(d_C);
}

//Транспонирование матрицы: C = A_T
template<typename T>
void MathCV::matrixTranspose(
    const T* A, T* C,
    size_t rows, size_t cols
){
    T* d_A = deviceAllocMatrix<T>(rows, cols);
    T* d_C = deviceAllocMatrix<T>(cols, rows);

    copyMatrixToDevice(A, d_A, rows, cols);

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (cols + blockSize.x - 1) / blockSize.x,
        (rows + blockSize.y - 1) / blockSize.y
    );

    matrixTransposeKernel<<<gridSize, blockSize>>>(d_A, d_C, rows, cols);
    copyMatrixToHost(d_C, C, cols, rows);

    freeDeviceMatrix(d_A);
    freeDeviceMatrix(d_C);
}

template<typename T>
T MathCV::computeDet(const T* h_A, size_t n) {
    T *d_A, *d_L, *d_U, *d_det;
    T h_det = 0.0;

    CHECK_CUDA_ERRORS(cudaMalloc(&d_A, n * n * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_L, n * n * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_U , n * n * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_det, sizeof(T)));

    CHECK_CUDA_ERRORS(cudaMemcpy(d_A, h_A, n * n * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemset(d_L, 0, n * n * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMemset(d_U, 0, n * n * sizeof(T)));

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 
                 (n + blockSize.y - 1) / blockSize.y);
    luDecompositionKernel<<<gridSize, blockSize>>>(d_A, d_L, d_U, n);

    determinantKernel<<<1, 1, sizeof(T)>>>(d_U, d_det, n);

    CHECK_CUDA_ERRORS(cudaMemcpy(&h_det, d_det, sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERRORS(cudaFree(d_A));
    CHECK_CUDA_ERRORS(cudaFree(d_L));
    CHECK_CUDA_ERRORS(cudaFree(d_U));
    CHECK_CUDA_ERRORS(cudaFree(d_det));

    return h_det;
}

template<typename T> 
void MathCV::inverseMatrixGPU(const T* d_A, T* d_invA, size_t n) {
    T* d_L = deviceAllocMatrix<T>(n, n);
    T* d_U = deviceAllocMatrix<T>(n, n);
    T* d_minors = deviceAllocMatrix<T>(n, n);
    T* d_adjugate = deviceAllocMatrix<T>(n, n);
    T* d_det;
    T h_det;

    CHECK_CUDA_ERRORS(cudaMalloc(&d_det, sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMemset(d_L, 0, n * n * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMemset(d_U, 0, n * n * sizeof(T)));

    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, 
                (n + blockSize.y - 1) / blockSize.y);

    T* d_A_nonconst = deviceAllocMatrix<T>(n, n);
    copyMatrixToDevice(d_A, d_A_nonconst, n, n);

    luDecompositionKernel<<<gridSize, blockSize>>>(d_A_nonconst, d_L, d_U, n);
    determinantKernel<<<1, 1>>>(d_U, d_det, n);

    CHECK_CUDA_ERRORS(cudaMemcpy(&h_det, d_det, sizeof(T), cudaMemcpyDeviceToHost));

    if (fabs(h_det) < 1e-10) {
        freeDeviceMatrix(d_L); freeDeviceMatrix(d_U); freeDeviceMatrix(d_det);
        freeDeviceMatrix(d_minors); freeDeviceMatrix(d_adjugate);
        freeDeviceMatrix(d_A_nonconst);
        throw std::runtime_error("Matrix is singular (determinant is zero)");
    }

    matrixOfMinorsKernel<<<gridSize, blockSize>>>(d_A_nonconst, d_minors, n);
    matrixTransposeKernel<<<gridSize, blockSize>>>(d_minors, d_adjugate, n, n);

    double invDet = 1.0 / static_cast<double>(h_det);
    scalarMultiplyKernel<<<gridSize, blockSize>>>(d_adjugate, invDet, n, n);

    copyMatrixToDevice(d_adjugate, d_invA, n, n);

    freeDeviceMatrix(d_L); 
    freeDeviceMatrix(d_U);
    freeDeviceMatrix(d_det);
    freeDeviceMatrix(d_minors);
    freeDeviceMatrix(d_adjugate);
    freeDeviceMatrix(d_A_nonconst);
}



// Explicit template instantiation
template int* MathCV::deviceAllocMatrix<int>(size_t, size_t);
template float* MathCV::deviceAllocMatrix<float>(size_t, size_t);
template double* MathCV::deviceAllocMatrix<double>(size_t, size_t);

template void MathCV::copyMatrixToDevice<int>(const int*, int*, size_t, size_t);
template void MathCV::copyMatrixToDevice<float>(const float*, float*, size_t, size_t);
template void MathCV::copyMatrixToDevice<double>(const double*, double*, size_t, size_t);

template void MathCV::copyMatrixToHost<int>(const int*, int*, size_t, size_t);
template void MathCV::copyMatrixToHost<float>(const float*, float*, size_t, size_t);
template void MathCV::copyMatrixToHost<double>(const double*, double*, size_t, size_t);

template void MathCV::freeDeviceMatrix<int>(int*);
template void MathCV::freeDeviceMatrix<float>(float*);
template void MathCV::freeDeviceMatrix<double>(double*);

template void MathCV::MatrixMul<int>(const int*, const int*, int*, size_t, size_t, size_t);
template void MathCV::MatrixMul<float>(const float*, const float*, float*, size_t, size_t, size_t);
template void MathCV::MatrixMul<double>(const double*, const double*, double*, size_t, size_t, size_t);

template void MathCV::matrixAdd<int>(const int*, const int*, int*, size_t, size_t);
template void MathCV::matrixAdd<float>(const float*, const float*, float*, size_t, size_t);
template void MathCV::matrixAdd<double>(const double*, const double*, double*, size_t, size_t);

template void MathCV::matrixSub<int>(const int*, const int*, int*, size_t, size_t);
template void MathCV::matrixSub<float>(const float*, const float*, float*, size_t, size_t);
template void MathCV::matrixSub<double>(const double*, const double*, double*, size_t, size_t);

template void MathCV::matrixTranspose<int>(const int*, int*, size_t, size_t);
template void MathCV::matrixTranspose<float>(const float*, float*, size_t, size_t);
template void MathCV::matrixTranspose<double>(const double*, double*, size_t, size_t);

template int MathCV::computeDet<int>(const int*, size_t);
template float MathCV::computeDet<float>(const float*, size_t);
template double MathCV::computeDet<double>(const double*, size_t);

template void MathCV::inverseMatrixGPU<int>(const int*, int*, size_t);
template void MathCV::inverseMatrixGPU<float>(const float*, float*, size_t);
template void MathCV::inverseMatrixGPU<double>(const double*, double*, size_t);

// Kernel instantiations
template __global__ void MathCV::matrixMulKernel<int>(const int*, const int*, int*, size_t, size_t, size_t);
template __global__ void MathCV::matrixMulKernel<float>(const float*, const float*, float*, size_t, size_t, size_t);
template __global__ void MathCV::matrixMulKernel<double>(const double*, const double*, double*, size_t, size_t, size_t);

template __global__ void MathCV::matrixAddKernel<int>(const int*, const int*, int*, size_t, size_t);
template __global__ void MathCV::matrixAddKernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void MathCV::matrixAddKernel<double>(const double*, const double*, double*, size_t, size_t);

template __global__ void MathCV::matrixSubKernel<int>(const int*, const int*, int*, size_t, size_t);
template __global__ void MathCV::matrixSubKernel<float>(const float*, const float*, float*, size_t, size_t);
template __global__ void MathCV::matrixSubKernel<double>(const double*, const double*, double*, size_t, size_t);

template __global__ void MathCV::matrixTransposeKernel<int>(const int*, int*, size_t, size_t);
template __global__ void MathCV::matrixTransposeKernel<float>(const float*, float*, size_t, size_t);
template __global__ void MathCV::matrixTransposeKernel<double>(const double*, double*, size_t, size_t);

template __global__ void MathCV::luDecompositionKernel<float>(float*, float*, float*, size_t);
template __global__ void MathCV::luDecompositionKernel<double>(double*, double*, double*, size_t);

template __global__ void MathCV::determinantKernel<float>(const float*, float*, size_t);
template __global__ void MathCV::determinantKernel<double>(const double*, double*, size_t);

template __global__ void MathCV::matrixOfMinorsKernel<int>(const int*, int*, size_t);
template __global__ void MathCV::matrixOfMinorsKernel<float>(const float*, float*, size_t);
template __global__ void MathCV::matrixOfMinorsKernel<double>(const double*, double*, size_t);

template __global__ void MathCV::scalarMultiplyKernel<int>(int*, double, size_t, size_t);
template __global__ void MathCV::scalarMultiplyKernel<double>(double*, double, size_t, size_t);
template __global__ void MathCV::scalarMultiplyKernel<float>(float*, double, size_t, size_t);

template __device__ int MathCV::computeSubmatrixDet<int>(const int*, size_t, int*);
template __device__ float MathCV::computeSubmatrixDet<float>(const float*, size_t, float*);
template __device__ double MathCV::computeSubmatrixDet<double>(const double*, size_t, double*);
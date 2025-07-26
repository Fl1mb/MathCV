#include "matrix_operations.cuh"

namespace MathCV {

// Умножение матриц
template<typename T>
void MatrixOperations<T>::multiply(
    const T* A, const T* B, T* C,
    size_t a_rows, size_t a_cols,
    size_t b_cols)
{
    if(CUDAChecker::isAvailable()) {
        multiplyCUDA(A, B, C, a_rows, a_cols, b_cols);
    } else {
        multiplyCPU(A, B, C, a_rows, a_cols, b_cols);
    }
}

// Сложение матриц
template<typename T>
void MatrixOperations<T>::summarize(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols)
{
    if(CUDAChecker::isAvailable()) {
        summarizeCUDA(A, B, C, rows, cols);
    } else {
        summarizeCPU(A, B, C, rows, cols);
    }
}

// Вычитание матриц
template<typename T>
void MatrixOperations<T>::subtraction(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols)
{
    if(CUDAChecker::isAvailable()) {
        subtractionCUDA(A, B, C, rows, cols);
    } else {
        subtractionCPU(A, B, C, rows, cols);
    }
}

// Транспонирование матрицы
template<typename T>
void MatrixOperations<T>::transpose(
    const T* A, T* A_T,
    size_t rows, size_t cols)
{
    if(CUDAChecker::isAvailable()) {
        transposeCUDA(A, A_T, rows, cols);
    } else {
        transposeCPU(A, A_T, rows, cols);
    }
}

// Вычисление определителя
template<typename T>
void MatrixOperations<T>::determinant(
    const T* A, double* det,
    size_t n)
{
    if(CUDAChecker::isAvailable()) {
        determinantCUDA(A, det, n);
    } else {
        determinantCPU(A, det, n);
    }
}

template<typename T>
void MatrixOperations<T>::inverse(
    const T* A, T* inv_A,
    size_t n
){
    if(CUDAChecker::isAvailable()){
        inverseCPU(A, inv_A, n);
    }else{
        inverseCUDA(A, inv_A, n);
    }
}

// =============================================
// Реализации для CPU
// =============================================

template<typename T>
void MatrixOperations<T>::multiplyCPU(
    const T* A, const T* B, T* C,
    size_t a_rows, size_t a_cols,
    size_t b_cols)
{
    for(size_t i = 0; i < a_rows; ++i) {
        for(size_t j = 0; j < b_cols; ++j) {
            T sum = 0;
            for(size_t k = 0; k < a_cols; ++k) {
                sum += A[i * a_cols + k] * B[k * b_cols + j];
            }
            C[i * b_cols + j] = sum;
        }
    }
}

template<typename T>
void MatrixOperations<T>::summarizeCPU(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols)
{
    for(size_t i = 0; i < rows * cols; ++i) {
        C[i] = A[i] + B[i];
    }
}

template<typename T>
void MatrixOperations<T>::subtractionCPU(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols)
{
    for(size_t i = 0; i < rows * cols; ++i) {
        C[i] = A[i] - B[i];
    }
}

template<typename T>
void MatrixOperations<T>::transposeCPU(
    const T* A, T* A_T,
    size_t rows, size_t cols)
{
    for(size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < cols; ++j) {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

template<typename T>
void MatrixOperations<T>::determinantCPU(
    const T* A, double* det,
    size_t n)
{
    // Простая реализация метода Гаусса для CPU
    std::vector<T> temp(n * n);
    std::copy(A, A + n * n, temp.begin());
    
    *det = 1;
    for(size_t i = 0; i < n; ++i) {
        // Поиск ведущего элемента
        size_t pivot = i;
        for(size_t j = i + 1; j < n; ++j) {
            if(std::abs(temp[j * n + i]) > std::abs(temp[pivot * n + i])) {
                pivot = j;
            }
        }
        
        if(pivot != i) {
            for(size_t j = 0; j < n; ++j) {
                std::swap(temp[i * n + j], temp[pivot * n + j]);
            }
            *det = -*det;
        }
        
        if(temp[i * n + i] == 0) {
            *det = 0;
            return;
        }
        
        *det *= temp[i * n + i];
        
        for(size_t j = i + 1; j < n; ++j) {
            T factor = temp[j * n + i] / temp[i * n + i];
            for(size_t k = i; k < n; ++k) {
                temp[j * n + k] -= factor * temp[i * n + k];
            }
        }
    }
}

template<typename T>
void MatrixOperations<T>::inverseCPU(
    const T* A, T* inv_A,
    size_t n
){
    double det;
    determinantCPU(A, &det, n);
    
    if(fabs(det) < 1e-10) {
        throw std::runtime_error("Matrix is singular (determinant is zero)");
    }

    std::vector<T> minors(n * n);
    std::vector<T> submatrix((n-1) * (n-1));

    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            size_t sub_idx = 0;
            for(size_t k = 0; k < n; ++k) {
                if(k == i) continue;
                for(size_t l = 0; l < n; ++l) {
                    if(l == j) continue;
                    submatrix[sub_idx++] = A[k * n + l];
                }
            }

            double sub_det;
            if(n == 2) {
                sub_det = submatrix[0];
            } else {
                determinantCPU(submatrix.data(), &sub_det, n-1);
            }

            minors[i * n + j] = ((i + j) % 2 == 0 ? 1 : -1) * sub_det;
        }
    }

    std::vector<T> adjugate(n * n);
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            adjugate[j * n + i] = minors[i * n + j];
        }
    }

    T inv_det = 1.0 / det;
    for(size_t i = 0; i < n * n; ++i) {
        inv_A[i] = adjugate[i] * inv_det;
    } 
}

// =============================================
// Реализации для GPU
// =============================================

template<typename T>
void MatrixOperations<T>::multiplyCUDA(
    const T* A, const T* B, T* C,
    size_t a_rows, size_t a_cols,
    size_t b_cols
){
    MatrixMul(A, B, C, a_rows, a_cols, b_cols);
}

template<typename T>
void MatrixOperations<T>::summarizeCUDA(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    matrixAdd(A, B, C, rows, cols);
}

template<typename T>
void MatrixOperations<T>::subtractionCUDA(
    const T* A, const T* B, T* C,
    size_t rows, size_t cols
){
    matrixSub(A, B, C, rows, cols);
}

template<typename T>
void MatrixOperations<T>::transposeCUDA(
    const T* A, T* A_T,
    size_t rows, size_t cols
){
    matrixTranspose(A, A_T, rows, cols);
}

template<typename T>
void MatrixOperations<T>::determinantCUDA(
    const T* A, double* det,
    size_t n
){
    *det = computeDet(A, n);
}

template<typename T>
void MatrixOperations<T>::inverseCUDA(
    const T* A, T* inv_A,
    size_t n
){
    inverseMatrixGPU(A, inv_A, n);
}

template class MatrixOperations<float>;
template class MatrixOperations<double>;
template class MatrixOperations<int>;
} // namespace MathCV
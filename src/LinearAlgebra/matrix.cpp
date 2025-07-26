#include "matrix.h"

using namespace MathCV;

template<typename T>
T& Matrix<T>::operator()(int row, int col){
    return data_[(row * columns_ + col)];
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const{
    if(this->rows() == other.rows() 
        && this->columns() == other.columns()){
            return std::memcmp(this->data_, other.data_, rows_ * columns_ * sizeof(T)) == 0;
    }
    return false;    
}

template<typename T>
const T& Matrix<T>::operator()(int row, int col) const{
    return data_[(row * columns_ + col)];
}

template<typename T>
Matrix<T>::Matrix(int rows, int cols): rows_(rows),  
    columns_(cols)
{
    data_ = new T[rows * cols];
    std::memset(data_, T(), rows_ * columns_ * sizeof(T));
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
    : rows_(other.rows_), columns_(other.columns_)
{
    data_ = new T[rows_ * columns_];
    std::memcpy(data_, other.data_, rows_ * columns_ * sizeof(T));
}

template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other) noexcept
    : rows_(other.rows_), columns_(other.columns_)
{
    data_ = new T[rows_ * columns_ ];
    std::memcpy(data_ , other.data_, rows_ * columns_ * sizeof(T));
    other.clear();
}

template<typename T>
void Matrix<T>::clear(){
    rows_ = 0;
    columns_ = 0;
    if(!data_)
        delete[] data_;
    data_ = nullptr;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other){
    if(*this == other){
        return *this;
    }
    rows_ = other.rows_;
    columns_ = other.columns_;
    std::memcpy(this->data_, other.data_, rows_ * columns_ * sizeof(T));
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) noexcept{
    if(*this == other){
        other.clear();
        return *this;
    }
    rows_ = other.rows_;
    columns_ = other.columns_;
    data_ = new T[rows_ * columns_];
    std::memcpy(data_, other.data_, rows_ * columns_ * sizeof(T));
    other.clear();
    return *this;
}
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const{
    Matrix<T> result(this->rows_, this->columns_);
    auto res = summarize(*this, other, result);
    if(res == Result::SUCCESS){return result;}
    else{
        throw MathExceptions(res);
    }
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other)const{
    Matrix<T> result;
    auto res = multiply(*this, other, result);
    if(res == Result::SUCCESS)return result;
    else{
        throw MathExceptions(res);
    }
}

template<typename T>
int Matrix<T>::size()const{
    return rows_ * columns_;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T& coef)const{
    Matrix<T> result(*this);
    for(auto i = 0; i < size(); ++i){
        result.data_[i] *= coef;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other)const{
    Matrix<T> result(this->rows_, this->columns_);
    auto res = substraction(*this, other, result);
    if(res == Result::SUCCESS){return result;}
    else{
        throw MathExceptions(res);
    }
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& coef){
    for(auto i = 0; i < size(); ++i){
        data_[i] *= coef;
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T& coef)const{
    return this->operator*(1 / coef);
}

template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T& coef){
    return this->operator*=(1 / coef);
}

template<typename T>
Result Matrix<T>::summarize(const Matrix<T>& first, const Matrix<T>& second, Matrix<T>& result){
    if(first.rows_ != second.rows_ || 
        first.columns_ != second.columns_){
            return Result::DIFFERENT_SIZE_MATRIX;
    }
    if(!first.data_ || !second.data_)return Result::FAILED;
    MatrixOperations<T>::summarize(first.data_, second.data_, result.data_, first.rows_, first.columns_);
    return Result::SUCCESS;
}
template<typename T>
Result Matrix<T>::substraction(const Matrix<T>& first, const Matrix<T>& second, Matrix<T>& result){
    if(first.rows_ != second.rows_ || 
        first.columns_ != second.columns_){
            return Result::DIFFERENT_SIZE_MATRIX;
    }
    if(!first.data_  || !second.data_)return Result::FAILED;
    MatrixOperations<T>::subtraction(first.data_, second.data_, result.data_, first.rows_, first.columns_);
    return Result::SUCCESS;
}
template<typename T>
Result Matrix<T>::multiply(const Matrix<T>& first, const Matrix<T>& second, Matrix<T>& result){
    if (first.columns_ != second.rows_) {
        return Result::INCOMPATIBLE_MATRIX_SIZE;
    }

    result = Matrix<T>(first.rows_, second.columns_);\

    MatrixOperations<T>::multiply(first.data_, second.data_, result.data_, first.rows_, first.columns_, second.columns_);
    return Result::SUCCESS;
}
template<typename T>
double Matrix<T>::det()const{
    double d = 0.0;
    auto res = determinant(*this, d);
    if(res == Result::SUCCESS)return d;
    else{
        throw MathExceptions(res);
    }
}
template<typename T>
double Matrix<T>::tr()const{
    double t;
    auto res = trace(*this, t);
    if(res == Result::SUCCESS)return t;
    else{
        throw MathExceptions(res);
    }
}
template<typename T>
Result Matrix<T>::determinant(const Matrix<T>& matrix, double& result){
    if (matrix.rows_ != matrix.columns_) {
        return Result::NOT_SQUARE_MATRIX;
    }

    if (matrix.rows_ == 1) {
        result = matrix.data_[0];
        return Result::SUCCESS;
    }

    if (matrix.rows_ == 2) {
        result = matrix(0,0) * matrix(1,1) - matrix(0,1) * matrix(1,0);
        return Result::SUCCESS;
    }
    MatrixOperations<T>::determinant(matrix.data_, &result, matrix.rows_);
    return Result::SUCCESS;
}
template<typename T>
Result Matrix<T>::inverse(const Matrix<T>& matrix, Matrix<T>& result){
    if(matrix.rows() != matrix.columns()){
        return Result::NOT_SQUARE_MATRIX;
    }
    try{
        MatrixOperations<T>::inverse(matrix.data_, result.data_, matrix.rows());
        return Result::SUCCESS;
    }
    catch(const std::exception& ex){
        std::cout << ex.what() << std::endl;
        return Result::FAILED;
    }
}
template<typename T>
Result Matrix<T>::transpose(const Matrix<T>& matrix, Matrix<T>& result){
    result = Matrix<T>(matrix.columns_, matrix.rows_);
    MatrixOperations<T>::transpose(matrix.data_, result.data_, matrix.rows_, matrix.columns_);
    return Result::SUCCESS;
}
template<typename T>
Result Matrix<T>::trace(const Matrix<T>& matrix, double& tr){
    if(matrix.rows_ != matrix.columns_){
        return Result::NOT_SQUARE_MATRIX;
    }
    tr = 0.0;
    for(int i = 0; i < matrix.rows_; ++i){
        tr += matrix(i,i);
    }
    return Result::SUCCESS;
}


template class Matrix<int>;
template class Matrix<double>;
template class Matrix<float>;
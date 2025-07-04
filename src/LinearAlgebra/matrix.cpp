#include "matrix.h"

double& MathCV::Matrix::operator()(int row, int col, int channel){
    return data_[(row * columns_ + col) * channels_ + channel];
}

const double& MathCV::Matrix::operator()(int row, int col, int channel) const{
    return data_[(row * columns_ + col) * channels_ + channel];
}

MathCV::Matrix::Matrix(const MathCV::Matrix& other)
    : rows_(other.rows_), columns_(other.columns_),
    channels_(other.channels_), data_(other.data_){}

MathCV::Matrix::Matrix(MathCV::Matrix&& other)
    : rows_(other.rows_), columns_(other.columns_),
    channels_(other.channels_), data_(other.data_)
{other.clear();}

void MathCV::Matrix::clear(){
    rows_ = 0;
    columns_ = 0;
    channels_ = 0;
    data_.clear();
}

MathCV::Matrix& MathCV::Matrix::operator=(const MathCV::Matrix& other){
    if(*this == other){
        return *this;
    }
    rows_ = other.rows_;
    columns_ = other.columns_;
    channels_ = other.channels_;
    data_.clear();
    data_ = other.data_;
    return *this;
}

MathCV::Matrix& MathCV::Matrix::operator=(MathCV::Matrix&& other){
    if(*this == other){
        other.clear();
        return *this;
    }
    rows_ = other.rows_;
    columns_ = other.columns_;
    channels_ = other.channels_;
    data_.clear();
    data_ = other.data_;
    other.clear();
    return *this;
}

MathCV::Matrix MathCV::Matrix::operator+(const Matrix& other) const{
    Matrix result;
    auto res = summarize(*this, other, result);
    if(res == Result::SUCCESS){return result;}
    else{
        throw MathCV::MathExceptions(res);
    }
}


MathCV::Matrix MathCV::Matrix::operator*(const Matrix& other)const{
    Matrix result;
    auto res = multiply(*this, other, result);
    if(res == Result::SUCCESS)return result;
    else{
        throw MathCV::MathExceptions(res);
    }
}

MathCV::Matrix MathCV::Matrix::operator*(const double& coef)const{
    Matrix result(*this);
    for(auto& iter : result.data_){
        iter *= coef;
    }
    return result;
}

MathCV::Matrix MathCV::Matrix::operator-(const Matrix& other)const{
    Matrix result;
    auto res = substraction(*this, other, result);
    if(res == Result::SUCCESS){return result;}
    else{
        throw MathCV::MathExceptions(res);
    }
}

MathCV::Matrix& MathCV::Matrix::operator*=(const double& coef){
    for(auto& iter : data_){
        iter *= coef;
    }
    return *this;
}

MathCV::Matrix MathCV::Matrix::operator/(const double& coef)const{
    return this->operator*(1 / coef);
}
    
MathCV::Matrix& MathCV::Matrix::operator/=(const double& coef){
    return this->operator*=(1 / coef);
}

MathCV::Result MathCV::Matrix::summarize(const Matrix& first, const Matrix& second, Matrix& result){
    if(first.rows_ != second.rows_ || 
        first.columns_ != second.columns_ 
        || first.channels_ != second.channels_){
            return Result::DIFFERENT_SIZE_MATRIX;
    }

    for(auto i = 0; i < first.total(); i++){
        result.data_[i] = first.data_[i] + second.data_[i];
    }
    return Result::SUCCESS;
}

MathCV::Result MathCV::Matrix::substraction(const Matrix& first, const Matrix& second, Matrix& result){
    if(first.rows_ != second.rows_ || 
        first.columns_ != second.columns_ 
        || first.channels_ != second.channels_){
            return Result::DIFFERENT_SIZE_MATRIX;
    }
    for(auto i = 0; i < first.total(); i++){
        result.data_[i] = first.data_[i] - second.data_[i];
    }
    return Result::SUCCESS;
}

MathCV::Result MathCV::Matrix::multiply(const Matrix& first, const Matrix& second, Matrix& result){
    if (first.columns_ != second.rows_) {
        return Result::INCOMPATIBLE_MATRIX_SIZE;
    }
    if (first.channels_ != second.channels_) {
        return Result::DIFFERENT_CHANNELS_COUNT;
    }

    result = Matrix(first.rows_, second.columns_, first.channels_);\

    for (int i = 0; i < first.rows_; ++i) {
        for (int j = 0; j < second.columns_; ++j) {
            for (int k = 0; k < first.columns_; ++k) {
                for (int c = 0; c < first.channels_; ++c) {
                    result(i, j, c) += first(i, k, c) * second(k, j, c);
                }
            }
        }
    }

    return Result::SUCCESS;
}

double MathCV::Matrix::det()const{
    double d = 0.0;
    auto res = determinant(*this, d);
    if(res == Result::SUCCESS)return d;
    else{
        throw MathCV::MathExceptions(res);
    }
}

double MathCV::Matrix::tr()const{
    double t;
    auto res = trace(*this, t);
    if(res == Result::SUCCESS)return t;
    else{
        throw MathCV::MathExceptions(res);
    }
}

MathCV::Result MathCV::Matrix::determinant(const Matrix& matrix, double& result){
    if (matrix.rows_ != matrix.columns_) {
        return Result::NOT_SQUARE_MATRIX;
    }
    if (matrix.channels_ != 1) {
        return Result::MULTICHANNEL_MATRIX;
    }

    if (matrix.rows_ == 1) {
        result = matrix.data_[0];
        return Result::SUCCESS;
    }

    if (matrix.rows_ == 2) {
        result = matrix(0,0) * matrix(1,1) - matrix(0,1) * matrix(1,0);
        return Result::SUCCESS;
    }
    
    result = 0.0;
    int sign = 1;
    for (int j = 0; j < matrix.columns_; ++j) {
        Matrix submatrix(matrix.rows_ - 1, matrix.columns_ - 1);
        
        for (int i = 1; i < matrix.rows_; ++i) {
            int colIndex = 0;
            for (int k = 0; k < matrix.columns_; ++k) {
                if (k == j) continue;
                submatrix(i-1, colIndex) = matrix(i, k);
                colIndex++;
            }
        }
        
        double subDet;
        Result res = determinant(submatrix, subDet);
        if (res != Result::SUCCESS) {
            return res;
        }
        
        result += sign * matrix(0, j) * subDet;
        sign = -sign;
    }
    return Result::SUCCESS;
}

MathCV::Result MathCV::Matrix::inverse(const Matrix& matrix, Matrix& result){
    double det = 0.0;
    auto res = determinant(matrix, det);
    if(res != Result::SUCCESS){
        return res; 
    }

    if (std::abs(det) < 1e-10) {
        return Result::SINGULAR_MATRIX;
    }

    if (matrix.rows_ == 1) {
        result = Matrix(1, 1);
        result(0, 0) = 1.0 / matrix(0, 0);
        return Result::SUCCESS;
    }

    if (matrix.rows_ == 2) {
        result = Matrix(2, 2);
        double invDet = 1.0 / det;
        
        result(0, 0) =  matrix(1, 1) * invDet;
        result(0, 1) = -matrix(0, 1) * invDet;
        result(1, 0) = -matrix(1, 0) * invDet;
        result(1, 1) =  matrix(0, 0) * invDet;
        
        return Result::SUCCESS;
    }

    result = Matrix(matrix.rows_, matrix.columns_);
    Matrix cofactors(matrix.rows_, matrix.columns_);


     for (int i = 0; i < matrix.rows_; ++i) {
        for (int j = 0; j < matrix.columns_; ++j) {
            Matrix minor(matrix.rows_ - 1, matrix.columns_ - 1);
            int minorRow = 0;
            for (int k = 0; k < matrix.rows_; ++k) {
                if (k == i) continue;
                
                int minorCol = 0;
                for (int l = 0; l < matrix.columns_; ++l) {
                    if (l == j) continue;
                    
                    minor(minorRow, minorCol) = matrix(k, l);
                    minorCol++;
                }
                minorRow++;
            }
            
            double minorDet;
            res = determinant(minor, minorDet);
            if (res != Result::SUCCESS) {
                return res;
            }
            
            cofactors(i, j) = ((i + j) % 2 == 0 ? 1 : -1) * minorDet;
        }
    }

    Matrix adjugate;
    res = transpose(cofactors, adjugate);
    if(res != Result::SUCCESS)return res;

    double invDet = 1.0 / det;
    for (int i = 0; i < result.rows_; ++i) {
        for (int j = 0; j < result.columns_; ++j) {
            result(i, j) = adjugate(i, j) * invDet;
        }
    }
    return Result::SUCCESS;
}

MathCV::Result MathCV::Matrix::transpose(const Matrix& matrix, Matrix& result){
    if (matrix.channels_ != 1) {
        result = Matrix(matrix.columns_, matrix.rows_, matrix.channels_);
        for(int c = 0; c < matrix.channels_; ++c) {
            for(int i = 0; i < matrix.rows_; ++i) {
                for(int j = 0; j < matrix.columns_; ++j) {
                    result(j, i, c) = matrix(i, j, c);
                }
            }
        }
        return Result::SUCCESS;
    }
    result = Matrix(matrix.columns_, matrix.rows_);

    for(int i = 0; i < matrix.rows_; ++i){
        for(int j = 0; j < matrix.columns_; ++j){
            result(i,j) = matrix(j,i);
        }
    }
    return Result::SUCCESS;
}

MathCV::Result MathCV::Matrix::trace(const Matrix& matrix, double& tr){
    if(matrix.rows_ != matrix.columns_){
        return Result::NOT_SQUARE_MATRIX;
    }
    if(matrix.channels_ != 1){
        return Result::MULTICHANNEL_MATRIX;
    }
    tr = 0.0;
    for(int i = 0; i < matrix.rows_; ++i){
        tr += matrix(i,i);
    }
    return Result::SUCCESS;
}

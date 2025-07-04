#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <cstring>
#include "math_enums.h"

namespace MathCV
{
    class Matrix;    
} 

class MathCV::Matrix{
public:
    Matrix() = default;
    Matrix(int rows, int cols, int channels = 1) \
        : rows_(rows), columns_(cols), channels_(channels),
          data_(rows_ * columns_ * channels_) {}
    Matrix(const Matrix& other);
    Matrix(Matrix&& other);
    Matrix& operator=(const Matrix& other);
    Matrix& operator=(Matrix&& other);

    bool operator==(const Matrix& other)const {return data_ == other.data_ && 
                                                rows_ == other.rows_ && 
                                                columns_ == other.columns_ &&
                                                channels_ == other.channels_;}
    bool operator!=(const Matrix& other)const{return !(*this == other);}
    double& operator()(int row, int col, int channel = 0);
    const double& operator()(int row, int col, int channel = 0) const;
    Matrix operator+(const Matrix& other)const;
    Matrix operator-(const Matrix& other)const;
    Matrix operator*(const Matrix& other)const;
    Matrix operator*(const double& coef)const;
    Matrix operator/(const double& coef)const;
    Matrix& operator/=(const double& coef);
    Matrix& operator*=(const double& coef);


    int rows()const noexcept{return rows_;}
    int columns()const noexcept{return columns_;}
    int channels()const noexcept{return channels_;}
    size_t total()const noexcept{return rows_ * columns_ * channels_;}
    double det()const;
    double tr()const;

    double* ptr() {return data_.data();}
    const double* ptr()const {return data_.data();}

    void clear();

    static Result summarize(const Matrix& first, const Matrix& second, Matrix& result);
    static Result substraction(const Matrix& first, const Matrix& second, Matrix& result);
    static Result multiply(const Matrix& first, const Matrix& other, Matrix& result);
    static Result inverse(const Matrix& matrix, Matrix& result);
    static Result determinant(const Matrix& matrix, double& result);
    static Result transpose(const Matrix& matrix, Matrix& result);
    static Result trace(const Matrix& matrix, double& tr);

private:
    int rows_;
    int columns_;
    int channels_;
    std::vector<double> data_;
};



#endif //MATRIX_H
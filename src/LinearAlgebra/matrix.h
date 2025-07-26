#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <cstring>
#include <concepts>
#include "math_enums.h"
#include "cuda/matrix_operations.cuh"

namespace MathCV
{
    template<typename T = double>
    class Matrix{
    public:
        static_assert(std::is_arithmetic<T>::value, 
                 "Matrix can only be instantiated with arithmetic types");
        Matrix():rows_(0), columns_(0), data_(nullptr){}
        Matrix(int rows, int cols);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        Matrix& operator=(const Matrix& other);
        Matrix& operator=(Matrix&& other) noexcept;

        bool operator==(const Matrix& other)const;
        bool operator!=(const Matrix& other)const{return !(*this == other);}
        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;
        Matrix operator+(const Matrix& other)const;
        Matrix operator-(const Matrix& other)const;
        Matrix operator*(const Matrix& other)const;
        Matrix operator*(const T& coef)const;
        Matrix operator/(const T& coef)const;
        Matrix& operator/=(const T& coef);
        Matrix& operator*=(const T& coef);


        int rows()const noexcept{return rows_;}
        int columns()const noexcept{return columns_;}
        size_t total()const noexcept{return rows_ * columns_;}
        double det()const;
        double tr()const;

        T* ptr() {return data_;}
        const T* ptr()const {return data_;}

        void clear();

        static Result summarize(const Matrix& first, const Matrix& second, Matrix& result);
        static Result substraction(const Matrix& first, const Matrix& second, Matrix& result);
        static Result multiply(const Matrix& first, const Matrix& other, Matrix& result);
        static Result inverse(const Matrix& matrix, Matrix& result);
        static Result determinant(const Matrix& matrix, double& result);
        static Result transpose(const Matrix& matrix, Matrix& result);
        static Result trace(const Matrix& matrix, double& tr);

    private:
        
    
        int size()const;

        int rows_;
        int columns_;
        T* data_;
    };

    
} 






#endif //MATRIX_H
#ifndef MATH_ENUMS_H
#define MATH_ENUMS_H

#include <stdexcept>

namespace MathCV{
    enum class Result : int{
        FAILED = 0,
        SUCCESS = 1,
        DIFFERENT_SIZE_MATRIX,
        INCOMPATIBLE_MATRIX_SIZE,
        DIFFERENT_CHANNELS_COUNT,
        NOT_SQUARE_MATRIX,
        MULTICHANNEL_MATRIX,
        SINGULAR_MATRIX,
    };


    class MathExceptions : std::exception{
    public:
        MathExceptions(Result res) : res_(res) {}
        const char* what()const noexcept override{
            switch (res_)
            {
            case Result::DIFFERENT_SIZE_MATRIX:
                return "Different matrix size";
            case Result::DIFFERENT_CHANNELS_COUNT:
                return "Different channels count";
            case Result::INCOMPATIBLE_MATRIX_SIZE:
                return "Incompatible matrix size";
            case Result::NOT_SQUARE_MATRIX:
                return "Matrix is not square";
            case Result::MULTICHANNEL_MATRIX:
                return "Multichannel matrix";
            case Result::SINGULAR_MATRIX:
                return "Matrix is singular";
            default:
                break;
            }
        }
        Result getResult() const {return res_;}
    private:
        Result res_;    
    };
}


#endif //MATH_ENUMS_H
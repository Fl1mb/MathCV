#ifndef VECTORND_H
#define VECTORND_H

#include <array>
#include <initializer_list>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace MathCV{
    template<size_t N>
    class VectorND;
}

template<size_t N>
class MathCV::VectorND{
private:
    static_assert(N > 0, "Vector dimension must be positive");
    std::array<double, N> data;
public:
    VectorND(){data.fill(0);}
    VectorND(std::initializer_list<double> init) {
        if (init.size() != N) {
            throw std::invalid_argument("Initializer list size mismatch");
        }
        std::copy(init.begin(), init.end(), data.begin());
    }
    double& operator[](size_t index) { return data.at(index); }
    const double& operator[](size_t index) const { return data.at(index); }

    Vector<N> operator+(const Vector<N>& other) const {
        Vector<N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }
    
    Vector<N> operator-(const Vector<N>& other) const {
        Vector<N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }
    
    Vector<N> operator*(double scalar) const {
        Vector<N> result;
        for (size_t i = 0; i < N; ++i) {
            result[i] = data[i] * scalar;
        }
        return result;
    }


    double dot(const Vector<N>& other) const {
        double result = 0.0;
        for (size_t i = 0; i < N; ++i) {
            result += data[i] * other[i];
        }
        return result;
    }

    double length() const {
        return std::sqrt(dot(*this));
    }
    
    Vector<N> normalized() const {
        double len = length();
        if (len < 1e-10) {
            throw std::runtime_error("Cannot normalize zero vector");
        }
        return *this * (1.0 / len);
    }
    
    static double distance(const Vector<N>& a, const Vector<N>& b) {
        return (a - b).length();
    }

    using Vector2D = Vector<2>;
    using Vector3D = Vector<3>;

    template<>
    Vector2D Vector<2>::rotate(double angle) const {
        double cos_a = std::cos(angle);
        double sin_a = std::sin(angle);
        return Vector2D{
            data[0] * cos_a - data[1] * sin_a,
            data[0] * sin_a + data[1] * cos_a
        };
    }

    template<>
    Vector3D Vector<3>::cross(const Vector3D& other) const {
        return Vector3D{
            data[1] * other[2] - data[2] * other[1],
            data[2] * other[0] - data[0] * other[2],
            data[0] * other[1] - data[1] * other[0]
        };
    }
};



#endif VECTORND_H
#include "vector2d.h"

using namespace MathCV;

Vector2D::Vector2D(Vector2D&& other) noexcept :
    x(other.x), y(other.y)
{
    other.x = 0;
    other.y = 0;
}

Vector2D::Vector2D(std::initializer_list<double> coords){
    if (coords.size() != 2) 
        throw std::invalid_argument("Vector2D requires exactly 2 coordinates");
    auto it = coords.begin();
    x = *it++;
    y = *it;
}

Vector2D& Vector2D::operator=(const Vector2D& other){
    x = other.x;
    y = other.y;
    return *this;
}

Vector2D& Vector2D::operator=(Vector2D&& other) noexcept{
    x = other.x;
    y = other.y;
    other.x = 0;
    other.y = 0;
    return *this;
}

bool Vector2D::operator==(const Vector2D& other)const{
    const double eps = 1e-10;
    return (std::abs(x - other.x) < eps) && (std::abs(y - other.y) < eps);
}

bool Vector2D::operator!=(const Vector2D& other)const{
    return !(*this == other);
}

bool Vector2D::isZero()const{
    const double eps = 1e-10;
    return (std::abs(x) < eps) && (std::abs(y) < eps);
}

bool Vector2D::isUnit()const{
    const double eps = 1e-10;
    return std::abs(length() - 1.0) < eps;
}

Vector2D Vector2D::operator+(const Vector2D& other) const {
    return Vector2D{x + other.x, y + other.y};
}

Vector2D Vector2D::operator-(const Vector2D& other) const {
    return Vector2D{x - other.x, y - other.y};
}

Vector2D Vector2D::operator*(double alpha)const {
    return Vector2D{x * alpha, y * alpha};
}

Vector2D Vector2D::operator*(const Vector2D& other)const {
    return Vector2D{x * other.x, y * other.y};
}

Vector2D Vector2D::rotate(double angle_rad)const{
    double cos_a = std::cos(angle_rad);
    double sin_a = std::sin(angle_rad);
    return {
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a
    };
}

double Vector2D::length() const {
    return std::sqrt(x * x + y * y);
}

double Vector2D::norm() const {
    return length();
}

double Vector2D::dot(const Vector2D& other) const{
    return x * other.x + y * other.y;
}


double Vector2D::distanceTo(const Vector2D& first, const Vector2D& second) {
    return (first - second).length();
}

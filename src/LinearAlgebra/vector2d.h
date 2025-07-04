#ifndef VECTORS_H
#define VECTORS_H

#include <initializer_list>
#include <stdexcept>
#include <exception>
#include <cmath>

namespace MathCV{
    class Vector2D;
}

class MathCV::Vector2D{
protected:
    double x;
    double y;
public:
    Vector2D() : x(0), y(0) {}
    Vector2D(Vector2D&& other) noexcept;
    Vector2D(std::initializer_list<double> coords);

    Vector2D& operator=(const Vector2D& other);
    Vector2D& operator=(Vector2D&& other) noexcept;

    virtual ~Vector2D()= default;

    bool operator==(const Vector2D& other)const;
    bool operator!=(const Vector2D& other)const;
    bool isZero()const;
    bool isUnit()const;

    Vector2D operator+(const Vector2D& other)const;
    Vector2D operator-(const Vector2D& other)const;
    Vector2D operator*(double alpha)const;
    Vector2D operator*(const Vector2D& other)const;
    Vector2D rotate(double angle_rad)const;

    double length() const;
    double norm() const;
    double dot(const Vector2D& other) const;
    double getX()const {return x;}
    double getY()const {return y;}

    void setX(double x_){x = x_;}
    void setY(double y_){y = y_;}
    
    static double distanceTo(const Vector2D& first, const Vector2D& second);
};



#endif VECTORS_H

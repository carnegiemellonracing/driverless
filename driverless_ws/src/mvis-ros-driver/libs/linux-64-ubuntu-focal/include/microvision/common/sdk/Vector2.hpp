//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 11, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/VectorN.hpp>
#include <microvision/common/sdk/Math.hpp>

#include <array>
#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Vector class for which can store 2 elements (x and y).
//!
//! Dedicated to be used for 2d calculations.
// ------------------------------------------------------------------------------
template<typename T>
class Vector2 : public VectorN<T, 2>
{
public: // type definitions
    using VectorBaseClass = VectorN<T, 2>;
    using ValueType       = typename VectorBaseClass::ValueType;
    using VectorData      = typename VectorBaseClass::VectorData;

    static constexpr uint8_t nbOfElements = VectorBaseClass::nbOfElements;
    using VectorBaseClass::operator*;

private:
    static constexpr uint8_t indexOfX{0};
    static constexpr uint8_t indexOfY{1};

public: // constructors
    //========================================
    //! \brief Empty default constructor.
    //!
    //! Initializes all elements to 0.
    //----------------------------------------
    Vector2() = default;

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] x  Initialization value for member x.
    //! \param[in] y  Initialization value for member y.
    //----------------------------------------
    Vector2(const ValueType x, const ValueType y) : VectorBaseClass({x, y}) {}

    //========================================
    //! \brief Copy constructor taking a vector of base class type.
    //! \param[in] vector  The matrix to be copied from.
    //----------------------------------------
    Vector2(const VectorData& vector) : VectorBaseClass(vector) {}

    //========================================
    //! \brief Copy constructor taking a vector of base class type.
    //! \param[in] vector  The matrix to be copied from.
    //----------------------------------------
    Vector2(const VectorBaseClass& src) : VectorBaseClass(src) {}

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~Vector2() override = default;

public: // operator
    //========================================
    //! \brief Operator for adding another Vector to this one.
    //! \param[in] other  The vector which shall be added to this one.
    //! \return A new vector holding  containing the sum of both vectors.
    //----------------------------------------
    Vector2<ValueType> operator+(const Vector2<ValueType>& other) const { return Vector2<ValueType>{*this} += other; }

    //========================================
    //! \brief Operator to subtract another vector from this vector.
    //! \param[in] other  The vector which shall be subtracted from this one.
    //! \return A new vector holding the difference of both vectors.
    //----------------------------------------
    Vector2<ValueType> operator-(const Vector2<ValueType>& other) const { return Vector2<ValueType>{*this} -= other; }

    //========================================
    //! \brief Dot product of this and the vector \a other.
    //! \param[in] other  The other vector to be multiplied by this vector.
    //! \return The dot product of this and \a other, i.e. the length of
    //!         the projection of this onto \a other or vise versa.
    //----------------------------------------
    ValueType operator*(const Vector2<ValueType> other) const
    {
        return (static_cast<const VectorN<ValueType, 2>&>(*this) * other);
    }

    //========================================
    //! \brief Operator for multiplying this Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return A new vector which is scaled about the factor.
    //----------------------------------------
    Vector2<ValueType> operator*(const ValueType factor) const { return Vector2<ValueType>{*this} *= factor; }

    //========================================
    //! \brief Operator for multiplying the Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return The product of this vector and the factor.
    //----------------------------------------
    friend Vector2<ValueType> operator*(const ValueType factor, const Vector2<ValueType> vector)
    {
        // multiplication with a scalar is commutative
        return Vector2<ValueType>{vector * factor};
    }

    //========================================
    //! \brief Divide this vector by a factor and return a copy.
    //! \param[in] divisor  The factor each vector element
    //!                     shall be divided by.
    //! \return The result of scaling this vector by the inverse divisor.
    //----------------------------------------
    Vector2<ValueType> operator/(const ValueType divisor) const { return Vector2<ValueType>{*this} /= divisor; }

    //========================================
    //! \brief Get the negative of \c this.
    //! \return The negative of \c this.
    //----------------------------------------
    Vector2<ValueType> operator-() const
    {
        Vector2<ValueType> negative; // 0
        negative -= *this;
        return negative;
    }

public: // math functions
    //========================================
    //! \brief Operator for calculating the cross product of two vectors.
    //! \param[in] other  The vector which shall used for the cross product calculation.
    //! \return The cross product of the two vectors.
    //! \sa operator^
    //----------------------------------------
    ValueType cross(const Vector2<ValueType>& other) const { return getX() * other.getY() - getY() * other.getX(); }

    //========================================
    //! \brief Calculates a normalized vector from this on.
    //! \return A new normalized vector.
    //! \sa normalize
    //----------------------------------------
    Vector2<ValueType> getNormalized() const { return Vector2<ValueType>{*this}.normalize(); }

    //========================================
    //! \brief Calculates a scaled version of this vector.
    //! \param[in] factor  The factor which will be used for scaling.
    //! \return A new vector holding a scaled version of this one.
    //! \sa scale
    //----------------------------------------
    Vector2<ValueType> getScaled(const ValueType factor) { return Vector2<ValueType>{*this} *= factor; }

public: //rotation functions
    //========================================
    //! \brief Calculates the angle between a vector facing in x direction and this one.
    //! \return The angle of the vector.
    //----------------------------------------
    ValueType getAngle() const { return std::atan2(getY(), getX()); }

    //========================================
    //! \brief Rotates this point around the origin (0,0).
    //! \param[in] angle  The angle about the vector will be rotated.
    //----------------------------------------
    void rotate(const ValueType angle) { *this = getRotated(angle); }

    //========================================
    //! \brief Calculates a rotated version of this vector.
    //! \param[in] angle  The angle about the vector will be rotated.
    //! \return A new vector which is rotated about the given angle.
    //----------------------------------------
    Vector2<ValueType> getRotated(const ValueType angle) const
    {
        const T rotationAngleCos{std::cos(angle)};
        const T rotationAngleSin{std::sin(angle)};

        return Vector2<T>{(getX() * rotationAngleCos) - (getY() * rotationAngleSin),
                          (getX() * rotationAngleSin) + (getY() * rotationAngleCos)};
    }

public: // member functions
    //========================================
    //! \brief Getter function for the x value, the first entry, of the vector.
    //! \return The x component of the vector.
    //----------------------------------------
    ValueType getX() const { return this->getValue(indexOfX); }

    //========================================
    //! \brief Setter function for the y value, the second entry, of the vector.
    //! \return The y component of the vector.
    //----------------------------------------
    ValueType getY() const { return this->getValue(indexOfY); }

    //========================================
    //! \brief Setter function for the x value, the first entry, of the vector.
    //! \param[in] val  The value which will replace the current x value.
    //----------------------------------------
    void setX(const ValueType val) { this->setValue(indexOfX, val); }

    //========================================
    //! \brief Setter function for the y value, the second entry, of the vector.
    //! \param[in] val  The value which will replace the current y value.
    //----------------------------------------
    void setY(const ValueType val) { this->setValue(indexOfY, val); }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Vector2<TT>& value);
    template<typename TT>
    friend void readLE(std::istream& is, Vector2<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Vector2<TT>& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const Vector2<TT>& value);

public:
    static constexpr bool isSerializable()
    {
        return (std::is_same<ValueType, float>{} || std::is_same<ValueType, int16_t>{}
                || std::is_same<ValueType, uint16_t>{} || std::is_same<ValueType, double>{});
    }

}; // Vector2

//==============================================================================
// Specializations for Stream operator
//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the vector content to a stream.
//! \param[in] os  The stream, the vector shall be written to.
//! \param[in] p   The vector which shall be streamed.
//! \return The stream to which the vector was written to.
//------------------------------------------------------------------------------
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector2<T>& p)
{
    os << "(" << p.getX() << ", " << p.getY() << ")";
    return os;
}

//==============================================================================
// Specializations for Serialization
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Vector2<T>& value)
{
    static_assert(Vector2<T>::isSerializable(), "writeBE is not implemented for given template type of Vector2");

    microvision::common::sdk::writeBE(os, value.m_elements[Vector2<T>::indexOfX]);
    microvision::common::sdk::writeBE(os, value.m_elements[Vector2<T>::indexOfY]);
}

//==============================================================================
template<typename T>
inline void readBE(std::istream& is, Vector2<T>& value)
{
    static_assert(Vector2<T>::isSerializable(), "readBE is not implemented for given template type of Vector2");

    microvision::common::sdk::readBE(is, value.m_elements[Vector2<T>::indexOfX]);
    microvision::common::sdk::readBE(is, value.m_elements[Vector2<T>::indexOfY]);
}

//==============================================================================
template<typename T>
inline void writeLE(std::ostream& os, const Vector2<T>& value)
{
    static_assert(Vector2<T>::isSerializable(), "writeLE is not implemented for given template type of Vector2");

    microvision::common::sdk::writeLE(os, value.m_elements[Vector2<T>::indexOfX]);
    microvision::common::sdk::writeLE(os, value.m_elements[Vector2<T>::indexOfY]);
}

//==============================================================================
template<typename T>
inline void readLE(std::istream& is, Vector2<T>& value)
{
    static_assert(Vector2<T>::isSerializable(), "readLE is not implemented for given template type of Vector2");

    microvision::common::sdk::readLE(is, value.m_elements[Vector2<T>::indexOfX]);
    microvision::common::sdk::readLE(is, value.m_elements[Vector2<T>::indexOfY]);
}

//==============================================================================
template<typename T>
inline constexpr std::streamsize serializedSize(const Vector2<T>&)
{
    static_assert(Vector2<T>::isSerializable(), "serializedSize is not implemented for given template type of Vector2");
    return Vector2<T>::nbOfElements * sizeof(T);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

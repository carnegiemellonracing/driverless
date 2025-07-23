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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

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
//! \brief Vector class for which can store 3 elements (x, y and z).
//!
//! Dedicated to be used for 3d calculations.
// ------------------------------------------------------------------------------
template<typename T>
class Vector3 : public VectorN<T, 3>
{
public: // type definitions
    using VectorBaseClass = VectorN<T, 3>;
    using ValueType       = typename VectorBaseClass::ValueType;
    using VectorData      = typename VectorBaseClass::VectorData;

    static constexpr uint8_t nbOfElements = VectorBaseClass::nbOfElements;

private:
    static constexpr uint8_t indexOfX{0};
    static constexpr uint8_t indexOfY{1};
    static constexpr uint8_t indexOfZ{2};

public: // constructors
    //========================================
    //! \brief Empty default constructor.
    //!
    //! Initializes all elements to 0.
    //----------------------------------------
    Vector3() = default;

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] x  Initialization value for member x.
    //! \param[in] y  Initialization value for member y.
    //! \param[in] z  Initialization value for member z.
    //----------------------------------------
    Vector3(const ValueType x, const ValueType y, const ValueType z) : VectorBaseClass({x, y, z}) {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] vector  Initialization value for the whole vector.
    //----------------------------------------
    Vector3(const VectorData& vector) : VectorBaseClass(vector) {}

    //========================================
    //! \brief Copy constructor taking a vector of base class type.
    //! \param[in] vector  The matrix to be copied from.
    //----------------------------------------
    Vector3(const VectorBaseClass& vector) : VectorBaseClass(vector) {}

    //========================================
    //! \brief Default Destructor
    //----------------------------------------
    ~Vector3() override = default;

public: // operators
    //========================================
    //! \brief Operator for adding another Vector to this one.
    //! \param[in] other  The vector which shall be added to this one.
    //! \return A new vector holding  containing the sum of both vectors.
    //----------------------------------------
    Vector3<ValueType> operator+(const Vector3<ValueType>& other) const { return Vector3<ValueType>{*this} += other; }

    //========================================
    //! \brief Operator to subtract another vector from this vector.
    //! \param[in] other  The vector which shall be subtracted from this one.
    //! \return A new vector holding the difference of both vectors.
    //----------------------------------------
    Vector3<ValueType> operator-(const Vector3<ValueType>& other) const { return Vector3<ValueType>{*this} -= other; }

    //========================================
    //! \brief Dot product of this and the vector \a other.
    //! \param[in] other  The other vector to be multiplied by this vector.
    //! \return The dot product of this and \a other, i.e. the length of
    //!         the projection of this onto \a other or vise versa.
    //----------------------------------------
    ValueType operator*(const Vector3<ValueType> other) const
    {
        return (static_cast<const VectorN<ValueType, 3>&>(*this) * other);
    }

    //========================================
    //! \brief Operator for multiplying this Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return A new vector which is scaled about the factor.
    //----------------------------------------
    Vector3<ValueType> operator*(const ValueType factor) const { return Vector3<ValueType>{*this} *= factor; }

    //========================================
    //! \brief Operator for multiplying the Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return The product of this vector and the factor.
    //----------------------------------------
    friend Vector3<ValueType> operator*(const ValueType factor, const Vector3<ValueType> vector)
    {
        // multiplication with a scalar is commutative
        return Vector3<ValueType>{vector * factor};
    }

    //========================================
    //! \brief Divide this vector by a factor and return a copy.
    //! \param[in] divisor  The factor each vector element
    //!                     shall be divided by.
    //! \return The result of scaling this vector by the inverse divisor.
    //----------------------------------------
    Vector3<ValueType> operator/(const ValueType divisor) const { return Vector3<ValueType>{*this} /= divisor; }

    //========================================
    //! \brief Get the negative of \c this.
    //! \return The negative of \c this.
    //----------------------------------------
    Vector3<ValueType> operator-() const
    {
        Vector3<ValueType> negative; // == 0
        negative -= *this;
        return negative;
    }

public: // math functions
        //========================================
    //! \brief Operator for calculating the cross product of two vectors.
    //! \param[in] other  The vector which shall used for the cross product calculation.
    //! \return The cross product of the two vectors.
    //----------------------------------------
    Vector3<ValueType> cross(const Vector3<ValueType>& other) const
    {
        return Vector3<ValueType>{(getY() * other.getZ()) - (getZ() * other.getY()),
                                  (getZ() * other.getX()) - (getX() * other.getZ()),
                                  (getX() * other.getY()) - (getY() * other.getX())};
    }

    //========================================
    //! \brief Calculates a normalized vector from this on.
    //! \return A new normalized vector.
    //! \sa normalize
    //----------------------------------------
    Vector3<ValueType> getNormalized() const { return Vector3<ValueType>{*this}.normalize(); }

    //========================================
    //! \brief Calculates a scaled version of this vector.
    //! \param[in] factor  The factor which will be used for scaling.
    //! \return A new vector holding a scaled version of this one.
    //! \sa scale
    //----------------------------------------
    Vector3<ValueType> getScaled(const ValueType factor) { return Vector3<ValueType>{*this} *= factor; }

public: //rotation functions
    //========================================
    //! \brief Calculates the rotation around the x-axis.
    //!
    //! This is the angle of 2 dimensional vector projected onto the y-z plane.
    //! \return The angle around the x-axis in [rad].
    //----------------------------------------
    ValueType getAngleAroundX() const { return std::atan2(getZ(), getY()); }

    //========================================
    //! \brief Calculates the rotation around the y-axis.
    //!
    //! This is the angle of 2 dimensional vector projected onto the x-z plane.
    //! \return The angle around the y-axis in [rad].
    //----------------------------------------
    ValueType getAngleAroundY() const { return std::atan2(getX(), getZ()); }

    //========================================
    //! \brief Calculates the rotation around the z-axis.
    //!
    //! This is the angle of 2 dimensional vector projected onto the x-y plane.
    //! \return The angle around the z-axis in [rad].
    //----------------------------------------
    ValueType getAngleAroundZ() const { return std::atan2(getY(), getX()); }

    //========================================
    //! \brief Rotates the vector around the x-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A reference to this after rotation.
    //----------------------------------------
    Vector3<ValueType>& rotateAroundX(const ValueType angle)
    {
        *this = getRotatedAroundX(angle);
        return *this;
    }

    //========================================
    //! \brief Rotates the vector around the x-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A new vector holding the rotated vector.
    //----------------------------------------
    Vector3<ValueType> getRotatedAroundX(const ValueType angle) const
    {
        const ValueType dCos = std::cos(angle);
        const ValueType dSin = std::sin(angle);

        const ValueType y = (getY() * dCos) - (getZ() * dSin);
        const ValueType z = (getZ() * dCos) + (getY() * dSin);

        return Vector3<ValueType>{getX(), y, z};
    }

    //========================================
    //! \brief Rotates the vector around the y-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A reference to this after rotation.
    //----------------------------------------
    Vector3<ValueType>& rotateAroundY(const ValueType angle)
    {
        *this = getRotatedAroundY(angle);
        return *this;
    }

    //========================================
    //! \brief Rotates the vector around the y-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A new vector holding the rotated vector.
    //----------------------------------------
    Vector3<ValueType> getRotatedAroundY(const ValueType angle) const
    {
        const ValueType dCos = std::cos(angle);
        const ValueType dSin = std::sin(angle);

        const ValueType x = (getX() * dCos) + (getZ() * dSin);
        const ValueType z = (getZ() * dCos) - (getX() * dSin);

        return Vector3<ValueType>{x, getY(), z};
    }

    //========================================
    //! \brief Rotates the vector around the z-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A reference to this after rotation.
    //----------------------------------------
    Vector3<ValueType>& rotateAroundZ(const ValueType angle)
    {
        *this = getRotatedAroundZ(angle);
        return *this;
    }

    //========================================
    //! \brief Rotates the vector around the z-axis.
    //! \param[in] angle  The angle in [rad] about the vector shall be rotated.
    //! \return A new vector holding the rotated vector.
    //----------------------------------------
    Vector3<ValueType> getRotatedAroundZ(const ValueType angle) const
    {
        const ValueType dCos = std::cos(angle);
        const ValueType dSin = std::sin(angle);

        const ValueType x = (getX() * dCos) - (getY() * dSin);
        const ValueType y = (getY() * dCos) + (getX() * dSin);

        return Vector3<ValueType>{x, y, getZ()};
    }

public: // member functions
    //========================================
    //! \brief Getter function for the x value.
    //! \return The x component of the vector.
    //----------------------------------------
    ValueType getX() const { return this->getValue(indexOfX); }

    //========================================
    //! \brief Getter function for the y value.
    //! \return The y component of the vector.
    //----------------------------------------
    ValueType getY() const { return this->getValue(indexOfY); }

    //========================================
    //! \brief Getter function for the z value.
    //! \return The z component of the vector.
    //----------------------------------------
    ValueType getZ() const { return this->getValue(indexOfZ); }

    //========================================
    //! \brief Setter function for the x value.
    //! \param[in] val  The new value for x.
    //----------------------------------------
    void setX(const ValueType val) { this->setValue(indexOfX, val); }

    //========================================
    //! \brief Setter function for the y value.
    //! \param[in] val  The new value for y.
    //----------------------------------------
    void setY(const ValueType val) { this->setValue(indexOfY, val); }

    //========================================
    //! \brief Setter function for the z value
    //! \param[in] val The  new value for z.
    //----------------------------------------
    void setZ(const ValueType val) { this->setValue(indexOfZ, val); }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Vector3<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Vector3<TT>& value);
    template<typename TT>
    friend void readLE(std::istream& is, Vector3<TT>& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const Vector3<TT>& value);

    template<typename TT>
    friend std::ostream& operator<<(std::ostream& os, const Vector3<TT>& value);
    template<typename TT>
    friend std::istream& operator>>(std::istream& is, Vector3<TT>& value);

public:
    static constexpr bool isSerializable()
    {
        return ((std::is_integral<ValueType>{} && std::is_signed<ValueType>{}) || std::is_floating_point<ValueType>{});
    }

}; // Vector3

//==============================================================================
// Specializations for stream operator.
//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the vector content to a stream.
//! \param[in, out] os  The stream, the vector shall be written to.
//! \param[in]      p   The vector which shall be streamed.
//! \return The stream to which the vector was written to.
//------------------------------------------------------------------------------
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Vector3<T>& value)
{
    os << "(" << value.m_elements[Vector3<T>::indexOfX] << "," << value.m_elements[Vector3<T>::indexOfY] << ","
       << value.m_elements[Vector3<T>::indexOfZ] << ")";
    return os;
}

//==============================================================================
//! \brief Stream operator for reading the vector content from a stream.
//! \param[in, out] is      The stream, the vector shall be read from.
//! \param[out]     value   The vector as read from the stream.
//! \return The stream from which the vector was read.
//!
//! \note If reading the data failed (check with /a istream::fail()) the content of the vector is undefined.
//------------------------------------------------------------------------------
template<typename T>
inline std::istream& operator>>(std::istream& is, Vector3<T>& value)
{
    char charBuffer;

    is.get(charBuffer);
    if (charBuffer != '(')
    {
        // Wrong prefix.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> value.m_elements[Vector3<T>::indexOfX];

    is.get(charBuffer);
    if (charBuffer != ',')
    {
        // Wrong separator.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> value.m_elements[Vector3<T>::indexOfY];

    is.get(charBuffer);
    if (charBuffer != ',')
    {
        // Wrong separator.
        is.setstate(std::ios::failbit);
        return is;
    }

    is >> value.m_elements[Vector3<T>::indexOfZ];

    is.get(charBuffer);
    if (charBuffer != ')')
    {
        // Wrong suffix.
        is.setstate(std::ios::failbit);
        return is;
    }

    return is;
}

//==============================================================================
// Specializations for serialization.
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Vector3<T>& value)
{
    static_assert(Vector3<T>::isSerializable(), "writeBE is not implemented for given template type of Vector3");

    microvision::common::sdk::writeBE(os, value.m_elements[Vector3<T>::indexOfX]);
    microvision::common::sdk::writeBE(os, value.m_elements[Vector3<T>::indexOfY]);
    microvision::common::sdk::writeBE(os, value.m_elements[Vector3<T>::indexOfZ]);
}

//==============================================================================
template<typename T>
inline void readBE(std::istream& is, Vector3<T>& value)
{
    static_assert(Vector3<T>::isSerializable(), "readBE is not implemented for given template type of Vector3");

    microvision::common::sdk::readBE(is, value.m_elements[Vector3<T>::indexOfX]);
    microvision::common::sdk::readBE(is, value.m_elements[Vector3<T>::indexOfY]);
    microvision::common::sdk::readBE(is, value.m_elements[Vector3<T>::indexOfZ]);
}
//==============================================================================

template<typename T>
inline void writeLE(std::ostream& os, const Vector3<T>& value)
{
    static_assert(Vector3<T>::isSerializable(), "writeLE is not implemented for given template type of Vector3");

    microvision::common::sdk::writeLE(os, value.m_elements[Vector3<T>::indexOfX]);
    microvision::common::sdk::writeLE(os, value.m_elements[Vector3<T>::indexOfY]);
    microvision::common::sdk::writeLE(os, value.m_elements[Vector3<T>::indexOfZ]);
}

//==============================================================================
template<typename T>
inline void readLE(std::istream& is, Vector3<T>& value)
{
    static_assert(Vector3<T>::isSerializable(), "readLE is not implemented for given template type of Vector3");

    microvision::common::sdk::readLE(is, value.m_elements[Vector3<T>::indexOfX]);
    microvision::common::sdk::readLE(is, value.m_elements[Vector3<T>::indexOfY]);
    microvision::common::sdk::readLE(is, value.m_elements[Vector3<T>::indexOfZ]);
}

//==============================================================================
template<typename T>
inline constexpr std::streamsize serializedSize(const Vector3<T>&)
{
    static_assert(Vector3<T>::isSerializable(), "serializedSize is not implemented for given template type of Vector3");
    return Vector3<T>::nbOfElements * sizeof(T);
}

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

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
//! \date Jan 22, 2020
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/io_prototypes.hpp>

#include <array>
#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Vector class for which can store N elements.
//!
//! Dedicated to be used for Nd calculations.
// ------------------------------------------------------------------------------
template<typename T, uint8_t n>
class VectorN
{
public: // type definitions
    using ValueType = T;

    static constexpr uint8_t nbOfElements{n};
    static constexpr uint8_t firstElement{0};

    using VectorData = std::array<ValueType, nbOfElements>;

public: // constructors
    //========================================
    //! \brief Empty default constructor.
    //!
    //! Initializes all elements to 0.
    //----------------------------------------
    VectorN() : m_elements{0} {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] vector  Initialization value for the whole vector.
    //----------------------------------------
    VectorN(const VectorData& vector) : m_elements{vector} {}

    //========================================
    //! \brief Default Destructor
    //----------------------------------------
    virtual ~VectorN() = default;

public: // operators
    //========================================
    //! \brief Operator for adding another VectorN to this one.
    //! \param[in] other  The vector which shall be added to this one.
    //! \return A reference to this vector then containing the sum of both vectors.
    //----------------------------------------
    VectorN<ValueType, n>& operator+=(const VectorN<ValueType, n>& other)
    {
        for (uint8_t iter = firstElement; iter < nbOfElements; ++iter)
        {
            m_elements[iter] += other.getValue(iter);
        }
        return *this;
    }

    //========================================
    //! \brief Operator to add another vector to this vector.
    //! \param[in] other  The vector which shall be added to this one.
    //! \return A new vector holding the sum of both vectors.
    //! \sa operator+=
    //----------------------------------------
    VectorN<ValueType, n> operator+(const VectorN<ValueType, n>& other) const
    {
        return VectorN<ValueType, n>{*this} += other;
    }

    //========================================
    //! \brief Operator to subtract another vector from this vector.
    //! \param[in] other  The vector which shall be subtracted from this one.
    //! \return A reference to this vector holding the difference of both vectors.
    //! \sa operator-
    //----------------------------------------
    VectorN<ValueType, n>& operator-=(const VectorN<ValueType, n>& other)
    {
        for (uint8_t iter = firstElement; iter < nbOfElements; ++iter)
        {
            m_elements[iter] -= other.getValue(iter);
        }
        return *this;
    }

    //========================================
    //! \brief Operator to subtract another vector from this vector.
    //! \param[in] other  The vector which shall be subtracted from this one.
    //! \return A new vector holding the difference of both vectors.
    //! \sa operator-=
    //----------------------------------------
    VectorN<ValueType, n> operator-(const VectorN<ValueType, n>& other) const
    {
        return VectorN<ValueType, n>{*this} -= other;
    }

    //========================================
    //! \brief Operator for multiplying the Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return A reference to this vector after scaling.
    //----------------------------------------
    VectorN<ValueType, n>& operator*=(const ValueType factor)
    {
        for (auto& element : m_elements)
        {
            element *= factor;
        }
        return *this;
    }

    //========================================
    //! \brief Operator for multiplying this Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return A new vector which is scaled about the factor.
    //----------------------------------------
    VectorN<ValueType, n> operator*(const ValueType factor) const { return VectorN<ValueType, n>{*this} *= factor; }

    //========================================
    //! \brief Operator for multiplying the Vector with a factor (scales the vector).
    //! \param[in] factor  The factor which shall be multiplied to the vector.
    //! \return The product of this vector and the factor.
    //----------------------------------------
    friend VectorN<ValueType, n> operator*(const ValueType factor, const VectorN<ValueType, n> vector)
    {
        // multiplication with a scalar is commutative
        return VectorN<ValueType, n>{vector * factor};
    }

    //========================================
    //! \brief Calculate the dot product of this and the \a other vector.
    //! \param[in] other The vector which shall be multiplied via dot product to the first one.
    //! \return A scalar value holding the result of the dot product.
    //----------------------------------------
    ValueType operator*(const VectorN<ValueType, n>& other) const
    {
        ValueType ret = 0;
        for (uint8_t iter = firstElement; iter < nbOfElements; ++iter)
        {
            ret += m_elements[iter] * other.getValue(iter);
        }
        return ret;
    }

    //========================================
    //! \brief Divide this vector by a factor.
    //! \param[in] divisor  The factor each vector element
    //!                     shall be divided by.
    //! \return Reference to this after scaling this vector.
    //! \sa scale
    //----------------------------------------
    VectorN<ValueType, n>& operator/=(const ValueType divisor)
    {
        for (auto& element : m_elements)
        {
            element /= divisor;
        }
        return *this;
    }

    //========================================
    //! \brief Divide this vector by a factor and return a copy.
    //! \param[in] divisor  The factor each vector element
    //!                     shall be divided by.
    //! \return The result of scaling this vector by the inverse divisor.
    //! \sa operator/=
    //! \sa getScaled
    //----------------------------------------
    VectorN<ValueType, n> operator/(const ValueType divisor) const { return VectorN<ValueType, n>{*this} /= divisor; }

    //========================================
    //! \brief Get the negative of \c this.
    //! \return The negative of \c this.
    //----------------------------------------
    VectorN<ValueType, n> operator-() const
    {
        VectorN<ValueType, n> negative; // == 0
        negative -= *this;
        return negative;
    }

public: // math functions
    //========================================
    //! \brief Calculates the length of the vector.
    //! \return The length of the vector.
    //----------------------------------------
    ValueType getLength() const { return std::sqrt(getSquaredLength()); }

    //========================================
    //! \brief Calculates the squared length of the vector.
    //! \return The squared length of the vector.
    //----------------------------------------
    ValueType getSquaredLength() const
    {
        ValueType ret = 0;
        for (auto& element : m_elements)
        {
            ret += element * element;
        }
        return ret;
    }

    //========================================
    //! \brief normalizes the vector to have a length of 1.
    //----------------------------------------
    VectorN<ValueType, n> normalize()
    {
        const T length{getLength()};
        if (!isZero(length))
        {
            *this /= length;
        }
        return *this;
    }

    //========================================
    //! \brief Calculates a normalized vector from this on.
    //! \return A new normalized vector.
    //! \sa normalize
    //----------------------------------------
    VectorN<ValueType, n> getNormalized() const { return VectorN<ValueType, n>{*this}.normalize(); }

    //========================================
    //! \brief Scales this vector by a factor.
    //! \param[in] factor  The factor which will be used for scaling.
    //! \sa operator*=
    //----------------------------------------
    void scale(const ValueType factor) { this->operator*=(factor); }

    //========================================
    //! \brief Calculates a scaled version of this vector.
    //! \param[in] factor  The factor which will be used for scaling.
    //! \return A new vector holding a scaled version of this one.
    //! \sa scale
    //----------------------------------------
    VectorN<ValueType, n> getScaled(const ValueType factor) { return VectorN<ValueType, n>{*this} *= factor; }

    //========================================
    //! \brief Checks this vector to have 0 length.
    //! \return \c True if the vector length is below epsilon (depends on the template type),
    //!         \c false otherwise.
    //----------------------------------------
    bool isZero() const { return isZero(getLength()); }

    //========================================
    //! \brief Checks whether the value is almost 0.
    //! \param[in] val  Value of floating point type to checked to be almost 0.
    //! \return \c True if the value is almost 0
    //!         \c false otherwise.
    //----------------------------------------
    template<typename FloatingType                                                           = ValueType,
             typename std::enable_if<std::is_floating_point<FloatingType>::value, int>::type = 0>
    static bool isZero(const ValueType val)
    {
        static_assert(std::is_same<ValueType, FloatingType>::value, "FloatingType and T have to be of the same type.");
        return (std::fabs(val) <= std::numeric_limits<ValueType>::epsilon());
    }

    //========================================
    //! \brief Checks whether the value is almost 0.
    //! \param[in] val  Value of integer type to checked to be almost 0.
    //! \return \c True if the value is almost 0
    //!         \c false otherwise.
    //----------------------------------------
    template<typename T_ = T, typename std::enable_if<std::is_integral<T_>::value, int>::type = 0>
    static bool isZero(const T val)
    {
        static_assert(std::is_same<T, T_>::value, "T_ and T have to be of the same type.");
        return (val == 0);
    }

    //========================================
    //! \brief Resets \c this vector to zero.
    //! \return A reference to \c this.
    //----------------------------------------
    VectorN<ValueType, n>& setToZero()
    {
        for (auto& element : m_elements)
        {
            element = 0;
        }
        return *this;
    }

public: // member functions
    // ========================================
    //! \brief Gets the value of the vector element for given \a index.
    //! \param[in] index  The index of the requested element. Counting starts at 0.
    //! \return The vector element value at the given \a index.
    //! \attention There is no bounds checking for \a index. To perform
    //!            a bounds checking use setValueChecked() instead.
    //! \sa getValueChecked()
    //----------------------------------------
    ValueType getValue(const uint8_t index) const { return m_elements[index]; }

    // ========================================
    //! \brief Gets the value of the vector element for given \a index.
    //! \param[in] index  The index of the requested element. Counting starts at 0.
    //! \return The vector element value at the given \a index.
    //! \attention Bounds checking for \a index will be performed. If you want to avoid
    //!            this due to performance reasons, use getValue() instead.
    //! \sa getValue()
    //----------------------------------------
    ValueType getValueChecked(const uint8_t index) const { return m_elements.at(index); }

    // ========================================
    //! \brief Sets the value of the vector element for given \a index.
    //! \param[in] index  The index of the requested element. Counting starts at 0.
    //! \param[in] value  The new value for the value entry at \a index.
    //! \attention There is no bounds checking for \a index. To perform
    //!            a bounds checking use setValueChecked() instead.
    //! \sa setValueChecked()
    //----------------------------------------
    void setValue(const uint8_t index, const ValueType value) { m_elements[index] = value; }

    // ========================================
    //! \brief Sets the value of the vector element for given \a index
    //! \param[in] index  The index of the requested element. Counting starts at 0.
    //! \param[in] value  The new value for the vector entry at \a index.
    //! \attention Bounds checking for \a index will be performed. If you want to avoid
    //!            this due to performance reasons, use setValue() instead.
    //! \sa setValue()
    //----------------------------------------
    void setValueChecked(const uint8_t index, const ValueType value) { m_elements.at(index) = value; }

protected: // member variables
    VectorData m_elements;

}; // VectorN

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Performs an entry-wise equality check.
//! \param[in] lhs  The first vector to be compared.
//! \param[in] rhs  The second vector to be compared.
//! \return \c True, if \c lhs and \c rhs are identically, \c false otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t n, typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
inline bool operator==(const VectorN<T, n>& lhs, const VectorN<T, n>& rhs)
{
    for (uint8_t iter = VectorN<T, n>::firstElement; iter < VectorN<T, n>::nbOfElements; ++iter)
    {
        if (!(lhs.getValue(iter) == rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise check, whether the \c lhs and  the \c rhs vector are not identically.
//! \param[in] lhs  The first vector to be compared.
//! \param[in] rhs  The second vector to be compared.
//! \return \c false if both vectors are identically, \c true otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t n>
inline bool operator!=(const VectorN<T, n>& lhs, const VectorN<T, n>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
//! \brief Performs an entry-wise equality check (special implementation for float).
//! \param[in] lhs  The first vector to be compared.
//! \param[in] rhs  The second vector to be compared.
//! \return \c True, if for each element of both vectors the difference is not smaller than 10^(-12) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n>
inline bool operator==(const VectorN<float, n>& lhs, const VectorN<float, n>& rhs)
{
    for (uint8_t iter = VectorN<float, n>::firstElement; iter < VectorN<float, n>::nbOfElements; ++iter)
    {
        if (fuzzyFloatUnequalT<12>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise equality check (special implementation for double).
//! \param[in] lhs  The first vector to be compared.
//! \param[in] rhs  The second vector to be compared.
//! \return \c True, if for each element of both vectors the difference is not smaller than 10^(-17) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n>
inline bool operator==(const VectorN<double, n>& lhs, const VectorN<double, n>& rhs)
{
    for (uint8_t iter = VectorN<double, n>::firstElement; iter < VectorN<double, n>::nbOfElements; ++iter)
    {
        if (fuzzyDoubleUnequalT<17>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//========================================
//! \brief Tests whether two vector values are nearly equal.
//! \tparam exp     The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] other  Vector to be compared to \c this.
//! \return \c True if the two vectors are equal in terms of the machine precision, which means their difference must
//!         be less than 10^(-exp). \c false otherwise.
//!
//! The exponent range is defined in negFloatPotenciesOf10.
//----------------------------------------
template<typename T, uint8_t n, uint8_t exp>
bool fuzzyCompareT(const VectorN<T, n>& lhs, const VectorN<T, n>& rhs)
{
    for (uint8_t iter = VectorN<T, n>::firstElement; iter < VectorN<T, n>::nbOfElements; ++iter)
    {
        if (!fuzzyCompareT<exp>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Fuzzy Compare two float vectors \a a and \a b. naN is equal naN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller than 10^(-exp) or if both are naN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const VectorN<float, 3>& lhs, const VectorN<float, 3>& rhs)
{
    for (uint8_t iter = VectorN<double, 3>::firstElement; iter < VectorN<double, 3>::nbOfElements; ++iter)
    {
        if (fuzzyFloatUnequalT<exp>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Fuzzy Compare two float vectors \a a and \a b. naN is equal naN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller than 10^(-exp) or if both are naN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const VectorN<float, 2>& lhs, const VectorN<float, 2>& rhs)
{
    for (uint8_t iter = VectorN<double, 2>::firstElement; iter < VectorN<double, 2>::nbOfElements; ++iter)
    {
        if (fuzzyFloatUnequalT<exp>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Fuzzy Compare two double vectors \a a and \a b. naN is equal naN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller than 10^(-exp) or if both are naN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const VectorN<double, 3>& lhs, const VectorN<double, 3>& rhs)
{
    for (uint8_t iter = VectorN<double, 3>::firstElement; iter < VectorN<double, 3>::nbOfElements; ++iter)
    {
        if (fuzzyDoubleUnequalT<exp>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
//! \brief Fuzzy Compare two double vectors \a a and \a b. naN is equal naN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First vector to be compared.
//! \param[in] rhs  Second vector to be compared.
//! \return \c True if the difference between \a a and \a b is not smaller than 10^(-exp) or if both are naN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const VectorN<double, 2>& lhs, const VectorN<double, 2>& rhs)
{
    for (uint8_t iter = VectorN<double, 2>::firstElement; iter < VectorN<double, 2>::nbOfElements; ++iter)
    {
        if (fuzzyDoubleUnequalT<exp>(lhs.getValue(iter), rhs.getValue(iter)))
        {
            return false;
        }
    }
    return true;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

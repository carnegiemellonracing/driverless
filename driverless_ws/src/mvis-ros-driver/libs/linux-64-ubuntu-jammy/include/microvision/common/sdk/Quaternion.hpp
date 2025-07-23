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
//! \date Jun 25, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/RotationMatrix3d.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//! \class Quaternion
//! \brief The quaternions are a number system that extends the complex numbers.
//! \f$x_0 + x_1 i + x_2 j + x_3 k \textrm{ with } i^2 = j^2 = k^2 = ijk = -1}f$
//!
//! Dedicated to be used for rotation calculations
// ------------------------------------------------------------------------------
template<typename ValueType, typename std::enable_if<std::is_floating_point<ValueType>::value, int>::type = 0>
class Quaternion final
{
public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Quaternion<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Quaternion<TT>& value);
    template<typename TT>
    friend void readLE(std::istream& is, Quaternion<TT>& value);
    template<typename TT>
    friend void writeLE(std::ostream& os, const Quaternion<TT>& value);

public: // type definitions
    //========================================

    using CoefficientArray = std::array<ValueType, 4>;

public:
    static constexpr bool isSerializable()
    {
        return ((std::is_integral<ValueType>{} && std::is_signed<ValueType>{}) || std::is_floating_point<ValueType>{});
    }

    static constexpr std::streamsize serializableSize{4 * sizeof(ValueType)};

public: // constructors
    //========================================
    //! \brief Default constructor.
    //!
    //! Initializes coefficient0/w to 1 and coefficient1/x, coefficient2/y and coefficient3/z to 0.
    //----------------------------------------
    Quaternion() : m_coefficientArray{{1, 0, 0, 0}} {}

    //========================================
    //! \brief Constructor with member initialization.
    //! \param[in] w  Initialization value for member coefficient0/w.
    //! \param[in] x  Initialization value for member coefficient1/x.
    //! \param[in] y  Initialization value for member coefficient2/y.
    //! \param[in] z  Initialization value for member coefficient3/z.
    //----------------------------------------
    Quaternion(const ValueType w, const ValueType x, const ValueType y, const ValueType z)
      : m_coefficientArray{{w, x, y, z}}
    {}

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    Quaternion(const Quaternion& other) : m_coefficientArray{other.m_coefficientArray}
    {
        // using "= default" is not working here since then
        // Quaternion<ValueType>{*this} will considered
        // would be considered as constant by visual studio.
    }

    Quaternion(const CoefficientArray& coeffs) : m_coefficientArray{coeffs} {}
    Quaternion& operator=(const Quaternion& other) = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Quaternion() {}

public: // operators
    //========================================
    //! \brief Operator for adding two quaternions.
    //! \param[in] other  The quaternion which shall be added to this one.
    //! \return A new quaternion holding the sum of both quaternions.
    //! \sa operator+=
    //----------------------------------------
    Quaternion<ValueType> operator+(const Quaternion<ValueType>& other) const
    {
        return Quaternion<ValueType>{m_coefficientArray[0] + other.m_coefficientArray[0], //
                                     m_coefficientArray[1] + other.m_coefficientArray[1], //
                                     m_coefficientArray[2] + other.m_coefficientArray[2], //
                                     m_coefficientArray[3] + other.m_coefficientArray[3]};
    }

    //========================================
    //! \brief Operator for adding another quaternion to this one.
    //! \param[in] other  The quaternion which shall be added to this one.
    //! \return The sum of both quaternions.
    //----------------------------------------
    Quaternion<ValueType>& operator+=(const Quaternion<ValueType>& other)
    {
        m_coefficientArray[0] += other.m_coefficientArray[0];
        m_coefficientArray[1] += other.m_coefficientArray[1];
        m_coefficientArray[2] += other.m_coefficientArray[2];
        m_coefficientArray[3] += other.m_coefficientArray[3];
        return *this;
    }

    //========================================
    //! \brief Operator for subtracting two quaternions.
    //! \param[in] other  The quaternion which shall be subtracted from this one.
    //! \return A new quaternion holding the difference of both quaternions.
    //! \sa operator-=
    //----------------------------------------
    Quaternion<ValueType> operator-(const Quaternion<ValueType>& other) const
    {
        return Quaternion<ValueType>{m_coefficientArray[0] - other.m_coefficientArray[0], //
                                     m_coefficientArray[1] - other.m_coefficientArray[1], //
                                     m_coefficientArray[2] - other.m_coefficientArray[2], //
                                     m_coefficientArray[3] - other.m_coefficientArray[3]};
    }

    //========================================
    //! \brief Operator to get the negativ of this quaternion.
    //! \return The negative of this quaternion.
    //----------------------------------------
    Quaternion<ValueType> operator-() const
    {
        return Quaternion<ValueType>{
            -m_coefficientArray[0], -m_coefficientArray[1], -m_coefficientArray[2], -m_coefficientArray[3]};
    }

    //========================================
    //! \brief Operator for subtracting another quaternion to this one.
    //! \param[in] other  The quaternion which shall be subtracting to this one.
    //! \return The difference of both quaternions.
    //----------------------------------------
    Quaternion<ValueType>& operator-=(const Quaternion<ValueType>& other)
    {
        m_coefficientArray[0] -= other.m_coefficientArray[0];
        m_coefficientArray[1] -= other.m_coefficientArray[1];
        m_coefficientArray[2] -= other.m_coefficientArray[2];
        m_coefficientArray[3] -= other.m_coefficientArray[3];
        return *this;
    }

    //========================================
    //! \brief Operator for multiplying the quaternion with a factor (scales the quaternion).
    //! \param[in] factor  The factor which shall be multiplied to the quaternion.
    //! \return A new quaternion which is scaled about the factor.
    //----------------------------------------
    Quaternion<ValueType> operator*(const ValueType factor) const
    {
        return Quaternion<ValueType>{m_coefficientArray[0] * factor, //
                                     m_coefficientArray[1] * factor, //
                                     m_coefficientArray[2] * factor, //
                                     m_coefficientArray[3] * factor};
    }

    //========================================
    //! \brief Operator for multiplying the quaternion with a factor (scales the quaternion).
    //! \param[in] factor  The factor which shall be multiplied to the quaternion.
    //! \return The scaled quaternion.
    //----------------------------------------
    Quaternion<ValueType>& operator*=(const ValueType factor)
    {
        m_coefficientArray[0] *= factor;
        m_coefficientArray[1] *= factor;
        m_coefficientArray[2] *= factor;
        m_coefficientArray[3] *= factor;
        return *this;
    }

    //========================================
    //! \brief Operator for dividing the quaternion by a divisor (scales the quaternion).
    //! \param[in] divisor  The divisor which shall be used for scaling the quaternion.
    //! \return A new quaternion holding a scaled quaternion.
    //! \sa operator/=
    //! \sa getScaled
    //----------------------------------------
    Quaternion<ValueType> operator/(const ValueType divisor) const
    {
        return Quaternion<ValueType>{m_coefficientArray[0] / divisor, //
                                     m_coefficientArray[1] / divisor, //
                                     m_coefficientArray[2] / divisor, //
                                     m_coefficientArray[3] / divisor};
    }

    //========================================
    //! \brief Operator for dividing the quaternion by a divisor (scales the quaternion).
    //! \param[in] divisor  The divisor which shall be used for scaling the quaternion.
    //! \return The scaled quaternion.
    //! \sa scale
    //----------------------------------------
    Quaternion<ValueType>& operator/=(const ValueType divisor)
    {
        m_coefficientArray[0] /= divisor;
        m_coefficientArray[1] /= divisor;
        m_coefficientArray[2] /= divisor;
        m_coefficientArray[3] /= divisor;
        return *this;
    }

    //========================================
    //! \brief Operator for getting the product of two quaternions.
    //! \param[in] other  The quaternion which shall be multiplied to the first one.
    //! \return A quaternion holding the result of the product.
    //----------------------------------------
    Quaternion<ValueType> operator*(const Quaternion<ValueType>& other) const
    {
        return Quaternion<ValueType>{
            m_coefficientArray[0] * other.m_coefficientArray[0] - m_coefficientArray[1] * other.m_coefficientArray[1]
                - m_coefficientArray[2] * other.m_coefficientArray[2]
                - m_coefficientArray[3] * other.m_coefficientArray[3], //
            m_coefficientArray[0] * other.m_coefficientArray[1] + m_coefficientArray[1] * other.m_coefficientArray[0]
                + m_coefficientArray[2] * other.m_coefficientArray[3]
                - m_coefficientArray[3] * other.m_coefficientArray[2], //
            m_coefficientArray[0] * other.m_coefficientArray[2] - m_coefficientArray[1] * other.m_coefficientArray[3]
                + m_coefficientArray[2] * other.m_coefficientArray[0]
                + m_coefficientArray[3] * other.m_coefficientArray[1], //
            m_coefficientArray[0] * other.m_coefficientArray[3] + m_coefficientArray[1] * other.m_coefficientArray[2]
                - m_coefficientArray[2] * other.m_coefficientArray[1]
                + m_coefficientArray[3] * other.m_coefficientArray[0]};
    }

    //========================================
    //! \brief Operator for comparing two quaternions for equality.
    //! \param[in] other  The quaternion which shall be compared to the first one.
    //! \return \c true if the difference between \a this and \a other is not smaller than 10^(-12) or if both are NaN.
    //!         \c false otherwise.
    //! \note This operator uses the fuzzyCompareT with a fixed epsilon 10^(-12).
    //!       For a variable epsilon please use fuzzyCompareT directly.
    //----------------------------------------
    bool operator==(const Quaternion<ValueType>& other) const
    {
        constexpr uint8_t fuzzyExponent{12};
        return fuzzyCompareT<ValueType, fuzzyExponent>(*this, other);
    }

    //========================================
    //! \brief Operator for comparing two quaternions for inequality.
    //! \param[in] other  The quaternion which shall be compared to the first one.
    //! \return False, if coefficient0, coefficient1, coefficient2 and coefficient3  of both quaternions are equal, true otherwise.
    //----------------------------------------
    bool operator!=(const Quaternion<ValueType>& other) const { return !(*this == other); }

public:
    //========================================
    //! \brief Operator for getting the dot product of two quaternions.
    //! \param[in] other  The quaternion which shall be multiplied to the first one.
    //! \return A scalar value holding the result of the dot product.
    //----------------------------------------
    ValueType dot(const Quaternion<ValueType>& other) const
    {
        return (m_coefficientArray[0] * other.m_coefficientArray[0]) //
               + (m_coefficientArray[1] * other.m_coefficientArray[1]) //
               + (m_coefficientArray[2] * other.m_coefficientArray[2]) //
               + (m_coefficientArray[3] * other.m_coefficientArray[3]);
    }

    //========================================
    //! \brief Operator for calculating the cross product of two quaternions.
    //! \param[in] other  The quaternion which shall used for the cross product calculation.
    //! \return The cross product of the two quaternions.
    //----------------------------------------
    Quaternion<ValueType> cross(const Quaternion<ValueType>& other) const
    {
        return Quaternion<ValueType>{0,
                                     (m_coefficientArray[2] * other.m_coefficientArray[3])
                                         - (m_coefficientArray[3] * other.m_coefficientArray[2]),
                                     (m_coefficientArray[3] * other.m_coefficientArray[1])
                                         - (m_coefficientArray[1] * other.m_coefficientArray[3]),
                                     (m_coefficientArray[1] * other.m_coefficientArray[2])
                                         - (m_coefficientArray[2] * other.m_coefficientArray[1])};
    }

    //========================================
    //! \brief Calculates the squared length of the quaternion.
    //! \return The squared length of the quaternion.
    //----------------------------------------
    ValueType getSquaredLength() const
    {
        return (m_coefficientArray[0] * m_coefficientArray[0]) //
               + (m_coefficientArray[1] * m_coefficientArray[1]) //
               + (m_coefficientArray[2] * m_coefficientArray[2]) //
               + (m_coefficientArray[3] * m_coefficientArray[3]);
    }

    //========================================
    //! \brief Calculates the length of the quaternion.
    //! \return The length of the quaternion.
    //----------------------------------------
    ValueType getLength() const { return std::sqrt(getSquaredLength()); }

    //========================================
    //! \brief Checks this quaternion to have 0 length.
    //! \return \c true, if length < epsilon, \c false otherwise. Epsilon depends on the template type.
    //----------------------------------------
    bool isZero() const
    {
        constexpr ValueType epsSquare
            = std::numeric_limits<ValueType>::epsilon() * std::numeric_limits<ValueType>::epsilon();

        // length is always positive
        // by comparing with square(eps) the square root of length has not to be calculated
        return (std::fabs(getSquaredLength()) <= epsSquare);
    }

    //========================================
    //! \brief Normalizes the Quaternion to have a length of 1.
    //! \return A reference to this to allow operation concatenation.
    //! \sa getNormalized
    //----------------------------------------
    Quaternion<ValueType>& normalize()
    {
        if (!isZero())
        {
            *this /= getLength();
        }
        return *this;
    }

    //========================================
    //! \brief Calculates a normalized quaternion from this on.
    //! \return A new normalized quaternion.
    //! \sa normalize
    //----------------------------------------
    Quaternion<ValueType> getNormalized() const { return Quaternion<ValueType>{*this}.normalize(); }

    //========================================
    //! \brief Inverts the quaternion.
    //!
    //! If the quaternion isZero, the quaternion will set exactly to 0.
    //!
    //! \return A reference to this to allow operation concatenation.
    //! \sa getInverse
    //----------------------------------------
    Quaternion<ValueType>& invert()
    {
        if (!isZero())
        {
            *this = getConjugated() / getSquaredLength();
        }
        else
        {
            // Set this Quaternion to exactly 0.
            m_coefficientArray[0] = 0;
            m_coefficientArray[1] = 0;
            m_coefficientArray[2] = 0;
            m_coefficientArray[3] = 0;
        }

        return *this;
    }

    //========================================
    //! \brief Calculates a inverse quaternion from this on.
    //!
    //! If the quaternion isZero, the quaternion will set exactly to 0.
    //!
    //! \return A new inverse quaternion.
    //! \sa invert
    //----------------------------------------
    Quaternion<ValueType> getInverse() const { return Quaternion<ValueType>{*this}.invert(); }

    //========================================
    //! \brief Conjungates the quaternion.
    //! \return A reference to this to allow operation concatenation.
    //! \sa getConjugated
    //----------------------------------------
    Quaternion<ValueType>& conjungate()
    {
        // m_coefficientArray[0] stays untouched
        m_coefficientArray[1] = -m_coefficientArray[1];
        m_coefficientArray[2] = -m_coefficientArray[2];
        m_coefficientArray[3] = -m_coefficientArray[3];
        return *this;
    }

    //========================================
    //! \brief Calculates the conjungate quaternion from this.
    //! \return A new conjungate quaternion.
    //! \sa conjungate
    //----------------------------------------
    Quaternion<ValueType> getConjugated() const { return Quaternion<ValueType>{*this}.conjungate(); }

public:
    //========================================
    //! \brief Calculates a rotation matrix from this on.
    //! \return A new rotation matix 3d.
    //----------------------------------------
    RotationMatrix3d<ValueType> getRotationMatrix() const
    {
        if (!isZero())
        {
            const Quaternion<ValueType> q = getNormalized();

            // short cuts for better maintainability
            const auto w = q.getW();
            const auto x = q.getX();
            const auto y = q.getY();
            const auto z = q.getZ();

            return RotationMatrix3d<ValueType>{1 - 2 * ((y * y) + (z * z)),
                                               2 * (-(w * z) + (x * y)),
                                               2 * (+(w * y) + (x * z)),

                                               2 * (+(w * z) + (x * y)),
                                               1 - 2 * ((x * x) + (z * z)),
                                               2 * (-(w * x) + (y * z)),

                                               2 * (-(w * y) + (x * z)),
                                               2 * (+(w * x) + (y * z)),
                                               1 - 2 * ((x * x) + (y * y))};
        }
        else
        {
            return RotationMatrix3d<ValueType>{}; // will be set to identity matrix in MatrixNxN
        }
    }

    //========================================
    //! \brief Calculates euler angels from this on.
    //! \return A new vector with the 3 angels (roll pitch, yaw).
    //----------------------------------------
    Vector3<ValueType> getEulerAngles() const
    {
        return this->getRotationMatrix().getEulerAnglesWithRotationOrderRollPitchYaw();
    }

    //========================================
    //! \brief Construct a quaternion from euler-angles.
    //!
    //! The rotation-axis order is (yaw, pitch, roll)
    //!
    //!  \param[in] euler vector including the euler angle
    //!  \return Quaternion, representing the rotation
    //----------------------------------------
    static Quaternion<ValueType> getQuaternionFromEulerAngle(const Vector3<ValueType>& euler)
    {
        RotationMatrix3d<ValueType> rot;
        return rot.setFromVectorWithRotationOrderRollPitchYaw(euler).getQuaternion();
    }

public: // getter
    //========================================
    //! \brief Getter function for the coefficient0/w value.
    //! \return The coefficient0/w component of the quaternion.
    //----------------------------------------
    ValueType getCoefficient0() const { return m_coefficientArray[0]; }

    //========================================
    //! \brief Getter function for the coefficient0/w value.
    //! \return The coefficient0/w component of the quaternion.
    //----------------------------------------
    ValueType getW() const { return m_coefficientArray[0]; }

    //========================================
    //! \brief Getter function for the coefficient1/x value.
    //! \return The coefficient1/x component of the quaternion.
    //----------------------------------------
    ValueType getCoefficient1() const { return m_coefficientArray[1]; }

    //========================================
    //! \brief Getter function for the coefficient1/x value.
    //! \return The coefficient1/x component of the quaternion.
    //----------------------------------------
    ValueType getX() const { return m_coefficientArray[1]; }

    //========================================
    //! \brief Getter function for the coefficient2/y value.
    //! \return The coefficient2/y component of the quaternion.
    //----------------------------------------
    ValueType getCoefficient2() const { return m_coefficientArray[2]; }

    //========================================
    //! \brief Getter function for the coefficient2/y value.
    //! \return The coefficient2/y component of the quaternion.
    //----------------------------------------
    ValueType getY() const { return m_coefficientArray[2]; }

    //========================================
    //! \brief Getter function for the coefficient3/z value.
    //! \return The coefficient3/z component of the quaternion.
    //----------------------------------------
    ValueType getCoefficient3() const { return m_coefficientArray[3]; }

    //========================================
    //! \brief Getter function for the coefficient3/z value.
    //! \return The coefficient3/z component of the quaternion.
    //----------------------------------------
    ValueType getZ() const { return m_coefficientArray[3]; }

    const CoefficientArray& getCoefficientArray() const { return m_coefficientArray; }

public: // setter
    //========================================
    //! \brief Setter function for the coefficient0/w value.
    //! \param[in] w  The value which will replace coefficient0/w.
    //----------------------------------------
    void setCoefficient0(const ValueType w) { m_coefficientArray[0] = w; }

    //========================================
    //! \brief Setter function for the coefficient0/w value.
    //! \param[in] w  The value which will replace coefficient0/w.
    //----------------------------------------
    void setW(const ValueType w) { m_coefficientArray[0] = w; }

    //========================================
    //! \brief Setter function for the coefficient1/x value.
    //! \param[in] x  The value which will replace coefficient1/x.
    //----------------------------------------
    void setCoefficient1(const ValueType x) { m_coefficientArray[1] = x; }

    //========================================
    //! \brief Setter function for the coefficient1/x value.
    //! \param[in] x  The value which will replace coefficient1/x.
    //----------------------------------------
    void setX(const ValueType x) { m_coefficientArray[1] = x; }

    //========================================
    //! \brief Setter function for the coefficient2/y value.
    //! \param[in] y  The value which will replace coefficient2/y.
    //----------------------------------------
    void setCoefficient2(const ValueType y) { m_coefficientArray[2] = y; }

    //========================================
    //! \brief Setter function for the coefficient2/y value.
    //! \param[in] y  The value which will replace coefficient2/y.
    //----------------------------------------
    void setY(const ValueType y) { m_coefficientArray[2] = y; }

    //========================================
    //! \brief Setter function for the coefficient3/z value.
    //! \param[in] z  The value which will replace coefficient3/z.
    //----------------------------------------
    void setCoefficient3(const ValueType z) { m_coefficientArray[3] = z; }

    //========================================
    //! \brief Setter function for the coefficient3/z value.
    //! \param[in] z  The value which will replace coefficient3/z.
    //----------------------------------------
    void setZ(const ValueType z) { m_coefficientArray[3] = z; }

    //========================================
    //! \brief Setter function for all coefficients.
    //! \param[in] w  The real coefficient of the quaternion.
    //! \param[in] x  The first imaginary coefficient (x) of the quaternion.
    //! \param[in] y  The second imaginary coefficient (y) of the quaternion.
    //! \param[in] z  The third imaginary coefficient (z) of the quaternion.
    //----------------------------------------
    void setValue(const ValueType w, const ValueType x, const ValueType y, const ValueType z)
    {
        m_coefficientArray[0] = w;
        m_coefficientArray[1] = x;
        m_coefficientArray[2] = y;
        m_coefficientArray[3] = z;
    }

    //========================================
    //! \brief Setter function for all coefficients.
    //! \param[in] coefficientArray  The new coefficients.
    //----------------------------------------
    void setValue(const CoefficientArray& coefficientArray) { m_coefficientArray = coefficientArray; }

protected: // member variables
    CoefficientArray m_coefficientArray;
}; // namespace sdk

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the vector content to a stream.
//! \param[in, out] os  The stream, the vector shall be written to.
//! \param[in]      q   The vector which shall be streamed.
//! \return The stream to which the vector was written to.
//------------------------------------------------------------------------------
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const Quaternion<T>& q)
{
    os << "(" << q.getW() << "," << q.getX() << "," << q.getY() << "," << q.getZ() << ")";
    return os;
}

//==============================================================================
//! \brief Tests whether two quaternion values are nearly equal.
//! \tparam exp     The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First quaternion to be compared.
//! \param[in] rhs  Second quaternion to be compared.
//! \return \c True if the two quaternions are equal in terms of the machine precision, which means their difference must
//!         be less than 10^(-exp). \c false otherwise.
//!
//! The exponent range is defined in negFloatPotenciesOf10.
//------------------------------------------------------------------------------
template<typename T, uint8_t exp>
bool fuzzyCompareT(const Quaternion<T>& lhs, const Quaternion<T>& rhs)
{
    auto lhsIter          = std::begin(lhs.getCoefficientArray());
    auto rhsIter          = std::begin(rhs.getCoefficientArray());
    const auto lhsEndIter = std::end(lhs.getCoefficientArray());

    for (; lhsIter != lhsEndIter; ++lhsIter, ++rhsIter)
    {
        if (!fuzzyCompareT<exp>(*lhsIter, *rhsIter))
        {
            return false;
        }
    }

    return true;
}

//==============================================================================
//! \brief Fuzzy Compare two double quaternions \a lhs and \a rhs. NaN is equal NaN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First quaternion to be compared.
//! \param[in] rhs  Second quaternion to be compared.
//! \return \c True if the difference between \a lhs and \a rhs is not smaller than 10^(-exp) or if both are NaN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const Quaternion<double>& lhs, const Quaternion<double>& rhs)
{
    auto lhsIter          = std::begin(lhs.getCoefficientArray());
    auto rhsIter          = std::begin(rhs.getCoefficientArray());
    const auto lhsEndIter = std::end(lhs.getCoefficientArray());

    for (; lhsIter != lhsEndIter; ++lhsIter, ++rhsIter)
    {
        if (!fuzzyDoubleEqualT<exp>(*lhsIter, *rhsIter))
        {
            return false;
        }
    }

    return true;
}
//==============================================================================
//! \brief Fuzzy Compare two double quaternions \a lhs and \a rhs. NaN is equal NaN here.
//! \tparam exp   The exponent of the epsilon used for the fuzzy compare. 10^(-exp).
//! \param[in] lhs  First quaternion to be compared.
//! \param[in] rhs  Second quaternion to be compared.
//! \return \c True if the difference between \a lhs and \a rhs is not smaller than 10^(-exp) or if both are NaN.
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t exp>
inline bool fuzzyEqualT(const Quaternion<float>& lhs, const Quaternion<float>& rhs)
{
    auto lhsIter          = std::begin(lhs.getCoefficientArray());
    auto rhsIter          = std::begin(rhs.getCoefficientArray());
    const auto lhsEndIter = std::end(lhs.getCoefficientArray());

    for (; lhsIter != lhsEndIter; ++lhsIter, ++rhsIter)
    {
        if (!fuzzyFloatEqualT<exp>(*lhsIter, *rhsIter))
        {
            return false;
        }
    }

    return true;
}

//==============================================================================
// specializations
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Quaternion<T>& q)
{
    static_assert(Quaternion<T>::isSerializable(), "writeBE is not implemented for given template type of Quaternion");

    for (const auto coeff : q.m_coefficientArray)
    {
        microvision::common::sdk::writeBE(os, coeff);
    }
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, Quaternion<T>& q)
{
    static_assert(Quaternion<T>::isSerializable(), "readBE is not implemented for given template type of Quaternion");

    for (auto& coeff : q.m_coefficientArray)
    {
        microvision::common::sdk::readBE(is, coeff);
    }
}

//==============================================================================

template<typename T>
inline void writeLE(std::ostream& os, const Quaternion<T>& q)
{
    static_assert(Quaternion<T>::isSerializable(), "writeLE is not implemented for given template type of Quaternion");

    for (const auto coeff : q.m_coefficientArray)
    {
        microvision::common::sdk::writeLE(os, coeff);
    }
}

//==============================================================================

template<typename T>
inline void readLE(std::istream& is, Quaternion<T>& q)
{
    static_assert(Quaternion<T>::isSerializable(), "readLE is not implemented for given template type of Quaternion");

    for (auto& coeff : q.m_coefficientArray)
    {
        microvision::common::sdk::readLE(is, coeff);
    }
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const Quaternion<T>&)
{
    static_assert(Quaternion<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of Quaternion");
    return Quaternion<T>::serializableSize;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

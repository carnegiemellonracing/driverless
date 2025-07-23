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

#include <microvision/common/sdk/Matrix3x3.hpp>

#include <type_traits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<typename FloatingType, typename std::enable_if<std::is_floating_point<FloatingType>::value, int>::type>
class Quaternion;

//==============================================================================
//! \brief Matrix class for which can store a 3x3 rotation matrix.
//!
//! Dedicated to be used for 3d calculations.
// ------------------------------------------------------------------------------
template<typename FloatingType, typename std::enable_if<std::is_floating_point<FloatingType>::value, int>::type = 0>
class RotationMatrix3d final : public Matrix3x3<FloatingType>
{
public:
    using MatrixBaseClass = MatrixNxN<FloatingType, 3>;

    using QuaternionType = Quaternion<FloatingType, 0>; // parameter 0 needed here due to template forward declaration

    static constexpr uint8_t nbOfRowsCols{MatrixBaseClass::nbOfRowsCols};

public: // constructors
    //========================================
    //! \brief Default constructors which initializes the matrix to an identity matrix.
    //----------------------------------------
    RotationMatrix3d() : Matrix3x3<FloatingType>() {}

    //========================================
    //! \brief Constructor for a rotation matrix with a rotation to align a vector to an other vector.
    //! \param[in] from  The vector to be rotated by the to be constructed rotation matrix to \a to.
    //! \param[in] to    The vector the \a from vector shall be rotated to by the to be constructed rotation matrix.
    //! \note See Tomas Moeller and John F. Hughes - Efficiently Building a Matrix to Rotate One Vector to Another.
    //----------------------------------------
    RotationMatrix3d(const Vector3<FloatingType>& from, const Vector3<FloatingType>& to) : Matrix3x3<FloatingType>()
    {
        if (to.isZero() || from.isZero()) // zero check
        {
            *this = RotationMatrix3d<FloatingType>{};
            return;
        }
        const Vector3<FloatingType> fromN = from.getNormalized();
        const Vector3<FloatingType> toN   = to.getNormalized();

        const FloatingType cosine = fromN * toN; // cosine of the angle

        constexpr FloatingType epsilon = static_cast<FloatingType>(0.000001);
        if (std::fabs(cosine) > 1 - epsilon) // "from" and "to"-vector almost parallel
        {
            // vector most nearly orthogonal to "from"
            Vector3<FloatingType> o{std::fabs(fromN.getX()), std::fabs(fromN.getY()), std::fabs(fromN.getZ())};
            if (o.getX() < o.getY())
            {
                if (o.getX() < o.getZ())
                {
                    o.setX(1);
                    o.setY(0);
                    o.setZ(0);
                }
                else
                {
                    o.setX(0);
                    o.setY(0);
                    o.setZ(1);
                }
            }
            else
            {
                if (o.getY() < o.getZ())
                {
                    o.setX(0);
                    o.setY(1);
                    o.setZ(0);
                }
                else
                {
                    o.setX(0);
                    o.setY(0);
                    o.setZ(1);
                }
            }

            // temporary storage vectors
            const Vector3<FloatingType> ut = o - fromN;
            const Vector3<FloatingType> vt = o - toN;

            // coefficients
            const FloatingType c1 = 2 / (ut * ut);
            const FloatingType c2 = 2 / (vt * vt);
            const FloatingType c3 = c1 * c2 * (ut * vt);

            RotationMatrix3d<FloatingType> rotationMatrix;
            for (uint8_t i = 0; i < nbOfRowsCols; ++i) // set rotation matrix
            {
                for (uint8_t j = 0; j < nbOfRowsCols; ++j)
                {
                    rotationMatrix.setValue(i,
                                            j,
                                            -(c1 * ut.getValue(i) * ut.getValue(j))
                                                - (c2 * vt.getValue(i) * vt.getValue(j))
                                                + (c3 * vt.getValue(i) * ut.getValue(j)));
                }
                rotationMatrix.setValue(i, i, rotationMatrix.getValue(i, i) + 1);
            }
            *this = rotationMatrix;
        }

        else // the most common case, unless "from"="to", or "from"=-"to"
        {
            const Vector3<FloatingType> fromToCrossProduct = fromN.cross(toN); // cross product
            const FloatingType h                           = 1 / (1 + cosine);
            const FloatingType hvx                         = h * fromToCrossProduct.getX();
            const FloatingType hvz                         = h * fromToCrossProduct.getZ();
            const FloatingType hvxy                        = hvx * fromToCrossProduct.getY();
            const FloatingType hvxz                        = hvx * fromToCrossProduct.getZ();
            const FloatingType hvyz                        = hvz * fromToCrossProduct.getY();

            RotationMatrix3d<FloatingType> rotationMatrix;

            rotationMatrix.setValue(0, 0, cosine + hvx * fromToCrossProduct.getX());
            rotationMatrix.setValue(0, 1, hvxy - fromToCrossProduct.getZ());
            rotationMatrix.setValue(0, 2, hvxz + fromToCrossProduct.getY());

            rotationMatrix.setValue(1, 0, hvxy + fromToCrossProduct.getZ());
            rotationMatrix.setValue(1, 1, cosine + h * fromToCrossProduct.getY() * fromToCrossProduct.getY());
            rotationMatrix.setValue(1, 2, hvyz - fromToCrossProduct.getX());

            rotationMatrix.setValue(2, 0, hvxz - fromToCrossProduct.getY());
            rotationMatrix.setValue(2, 1, hvyz + fromToCrossProduct.getX());
            rotationMatrix.setValue(2, 2, cosine + hvz * fromToCrossProduct.getZ());

            *this = rotationMatrix;
        }
    }

    //========================================
    //! \brief Inherit the constructors from Matrix3x3.
    //----------------------------------------
    using Matrix3x3<FloatingType>::Matrix3x3;

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    RotationMatrix3d(const Matrix3x3<FloatingType>& matrix) : Matrix3x3<FloatingType>(matrix) {}

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    RotationMatrix3d(const MatrixBaseClass& matrix) : Matrix3x3<FloatingType>(matrix) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RotationMatrix3d() final = default;

public:
    //========================================
    //! \brief Checks whether this RotationMatrix3d actually is a rotation matrix.
    //!
    //! \note Method exists only for FloatingType = float.
    //!
    //! \tparam S  is by default = FloatingType.
    //! \return Returns, whether transposed(A) * A =  A * transposed(A) = I and det = 1
    //----------------------------------------
    template<typename S                                                         = FloatingType, //
             typename std::enable_if<std::is_same<S, float>::value, bool>::type = true>
    bool isValid() const
    {
        const RotationMatrix3d<S> identity;
        const uint8_t precisionExponent{5};
        return fuzzyFloatEqualT<nbOfRowsCols, precisionExponent>((*this) * this->getTransposed(), identity)
               && fuzzyFloatEqualT<nbOfRowsCols, precisionExponent>(this->getTransposed() * (*this), identity)
               && fuzzyFloatEqualT<precisionExponent>(this->getDeterminant(), 1);
    }

    //========================================
    //! \brief Checks whether this RotationMatrix3d actually is a rotation matrix.
    //!
    //! \note Method exists only for FloatingType = double.
    //!
    //! \tparam S  is by default = FloatingType.
    //! \return Returns, whether transposed(A) * A =  A * transposed(A) = I and det = 1
    //----------------------------------------
    template<typename S                                                         = FloatingType, //
             typename std::enable_if<std::is_same<S, double>::value, int>::type = true>
    bool isValid() const
    {
        const RotationMatrix3d<S> identity;
        const uint8_t precisionExponent{5};
        return fuzzyDoubleEqualT<nbOfRowsCols, precisionExponent>((*this) * this->getTransposed(), identity)
               && fuzzyDoubleEqualT<nbOfRowsCols, precisionExponent>(this->getTransposed() * (*this), identity)
               && fuzzyDoubleEqualT<precisionExponent>(this->getDeterminant(), 1);
    }

public:
    //========================================
    //! \brief Sets the matrix to a rotation matrix around the x-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A reference to this.
    //----------------------------------------
    RotationMatrix3d<FloatingType>& setToRotationAroundX(const FloatingType angle)
    {
        *this = getRotationAroundX(angle);
        return *this;
    }

    //========================================
    //! \brief Gets a rotation matrix of a rotation around the x-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A new matrix holding the rotated matrix.
    //----------------------------------------
    static RotationMatrix3d<FloatingType> getRotationAroundX(const FloatingType angle)
    {
        const FloatingType c = std::cos(angle);
        const FloatingType s = std::sin(angle);
        return Matrix3x3<FloatingType>{1, 0, 0, 0, c, -s, 0, s, c};
    }

    //========================================
    //! \brief Sets the matrix to a rotation matrix around the y-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A reference to this.
    //----------------------------------------
    RotationMatrix3d<FloatingType>& setToRotationAroundY(const FloatingType angle)
    {
        *this = getRotationAroundY(angle);
        return *this;
    }

    //========================================
    //! \brief Gets a rotation matrix of a rotation around the y-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A new matrix holding the rotated matrix.
    //----------------------------------------
    static RotationMatrix3d<FloatingType> getRotationAroundY(const FloatingType angle)
    {
        const FloatingType c = std::cos(angle);
        const FloatingType s = std::sin(angle);
        return Matrix3x3<FloatingType>{c, 0, s, 0, 1, 0, -s, 0, c};
    }

    //========================================
    //! \brief Sets the matrix to a rotation matrix around the z-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A reference to this.
    //----------------------------------------
    RotationMatrix3d<FloatingType>& setToRotationAroundZ(const FloatingType angle)
    {
        *this = getRotationAroundZ(angle);
        return *this;
    }

    //========================================
    //! \brief Gets a rotation matrix of a rotation around the z-axis.
    //! \param[in] angle  The angle in [rad] for the rotation.
    //! \return A new matrix holding the rotated matrix.
    //----------------------------------------
    static RotationMatrix3d<FloatingType> getRotationAroundZ(const FloatingType angle)
    {
        const FloatingType c = std::cos(angle);
        const FloatingType s = std::sin(angle);
        return Matrix3x3<FloatingType>{c, -s, 0, s, c, 0, 0, 0, 1};
    }

    //========================================
    //! \brief Get euler angels from the rotation matrix for the
    //!        rotation order roll pitch yaw (matrix multiplication order yaw pitch roll).
    //! \return A new vector holding the euler angels.
    //----------------------------------------
    Vector3<FloatingType> getEulerAnglesWithRotationOrderRollPitchYaw() const
    {
        const FloatingType a = Vector2<FloatingType>{this->getValue(0, 0), this->getValue(1, 0)}.getLength();

        if (!fuzzyCompareT<6>(a, 0))
        {
            const FloatingType x = std::atan2(this->getValue(2, 1), this->getValue(2, 2));
            const FloatingType y = std::atan2(-this->getValue(2, 0), a);
            const FloatingType z = std::atan2(this->getValue(1, 0), this->getValue(0, 0));
            return Vector3<FloatingType>{x, y, z};
        }
        else
        {
            const FloatingType x = std::atan2(-this->getValue(1, 2), this->getValue(1, 1));
            const FloatingType y = std::atan2(-this->getValue(2, 0), a);
            const FloatingType z = 0;
            return Vector3<FloatingType>{x, y, z};
        }
    }

    //========================================
    //! \brief Set the rotation matrix from euler angels for the rotation order roll pitch yaw (multiplication order yaw pitch roll).
    //! \param[in] rotation  Vector including the euler angel in [rad].
    //! \return A new matrix holding the rotated matrix.
    //----------------------------------------
    RotationMatrix3d<FloatingType>& setFromVectorWithRotationOrderRollPitchYaw(const Vector3<FloatingType>& rotation)
    {
        RotationMatrix3d<FloatingType> rotMx;
        rotMx.setToRotationAroundX(rotation.getX());
        RotationMatrix3d<FloatingType> rotMy;
        rotMy.setToRotationAroundY(rotation.getY());
        RotationMatrix3d<FloatingType> rotMz;
        rotMz.setToRotationAroundZ(rotation.getZ());

        // Combined rotation matrix
        *this = RotationMatrix3d<FloatingType>((rotMz * rotMy) * rotMx);
        return *this;
    }

    //========================================
    //! \brief Get the quaternion corresponding to the rotation matrix.
    //! \return A quaternion holding the rotation.
    //! \note "Accurate Computation of Quaternions from Rotation Matrices",
    //!        Soheil Sarabandi and Federico Thomas
    //!        http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf
    //----------------------------------------
    QuaternionType getQuaternion() const
    {
        // short cuts for better maintainability
        const auto& elements = Matrix3x3<FloatingType>::getMatrix();

        const auto m00 = elements[0][0];
        const auto m01 = elements[0][1];
        const auto m02 = elements[0][2];

        const auto m10 = elements[1][0];
        const auto m11 = elements[1][1];
        const auto m12 = elements[1][2];

        const auto m20 = elements[2][0];
        const auto m21 = elements[2][1];
        const auto m22 = elements[2][2];

        FloatingType t; // value of the trace +1
        FloatingType w; // unscaled w/coeff0 value
        FloatingType x; // unscaled x/coeff1 value
        FloatingType y; // unscaled y/coeff2 value
        FloatingType z; // unscaled z/coeff3 value

        if (m22 < 0)
        {
            if (m00 >= m11)
            { // trace is <= 0 and m00 is dominant
                w = m21 - m12; // -1*s
                x = 1 + (m00 - m11) - m22; // *2, ^2
                y = m01 + m10; // *s
                z = m02 + m20; // *s

                t = x;
            }
            else
            { // trace is <= 0 and m11 is dominant
                w = m02 - m20;
                x = m01 + m10;
                y = 1 - m00 + m11 - m22;
                z = m12 + m21;

                t = y;
            }
        }
        else
        {
            if (m00 < -m11)
            { // trace is <= 0 and m22 is dominant
                w = m10 - m01;
                x = m02 + m20;
                y = m12 + m21;
                z = 1 - m00 - m11 + m22;

                t = z;
            }
            else
            { // trace > 0
                w = 1 + m00 + m11 + m22;
                x = m21 - m12;
                y = m02 - m20;
                z = m10 - m01;

                t = w;
            }
        }

        // scaling factor to be multiplied to the unscaled coefficient values
        const FloatingType f = static_cast<FloatingType>(0.5) / std::sqrt(t);
        return QuaternionType(f * w, f * x, f * y, f * z);
    }

public: // overloaded functions
    //========================================
    //! \brief Inverts the matrix.
    //! \return A reference to this after inverting.
    //----------------------------------------
    RotationMatrix3d<FloatingType>& invert() final
    {
        *this = getInverse();
        return *this;
    }

    //========================================
    //! \brief Calculates the inverse of the matrix.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    RotationMatrix3d<FloatingType> getInverse() const { return this->getTransposed(); }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, RotationMatrix3d<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const RotationMatrix3d<TT>& value);

}; // RotationMatrix3d

//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const RotationMatrix3d<T>& p)
{
    static_assert(RotationMatrix3d<T>::isSerializable(),
                  "writeBE is not implemented for given template type of RotationMatrix3d");

    for (uint8_t i = 0; i < 3; ++i)
    {
        for (uint8_t j = 0; j < 3; ++j)
        {
            microvision::common::sdk::writeBE(os, p.getValue(i, j));
        }
    }
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, RotationMatrix3d<T>& p)
{
    static_assert(RotationMatrix3d<T>::isSerializable(),
                  "readBE is not implemented for given template type of RotationMatrix3d");

    for (uint8_t i = 0; i < 3; ++i)
    {
        for (uint8_t j = 0; j < 3; ++j)
        {
            T value = 0;
            microvision::common::sdk::readBE(is, value);
            p.setValue(i, j, value);
        }
    }
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const RotationMatrix3d<T>&)
{
    static_assert(RotationMatrix3d<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of RotationMatrix3d");
    return RotationMatrix3d<T>::nbOfRowsCols * RotationMatrix3d<T>::nbOfRowsCols * sizeof(T);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

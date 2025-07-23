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

#include <microvision/common/sdk/RotationMatrix2d.hpp>
#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================
//! \brief Transformation matrix class which holds a 2x2 rotation matrix and
//! a 2x1 dimensional position vector. The size of the matrix is 3x3.
//! The structure is:
//!
//! | rotM posV |
//! | 0  0    1 |
//!
//! Dedicated to be used for 2d coordinate transformations
// ------------------------------------------------------------------------------
template<typename T>
class TransformationMatrix2d final
{
public:
    using ValueType = T;

public: // constructors
    //========================================
    //! \brief Default constructors which initializes the matrix to an identity
    //! matrix
    //----------------------------------------
    TransformationMatrix2d() : m_rotM{RotationMatrix2d<ValueType>{}}, m_posV{Vector2<ValueType>{}} {}

    //========================================
    //! \brief Constructor with rotation matrix initialization
    //! \param[in] rm The rotation matrix which shall be used for initialization
    //----------------------------------------
    TransformationMatrix2d(const RotationMatrix2d<ValueType>& rm) : m_rotM{rm}, m_posV{Vector2<ValueType>{}} {}

    //========================================
    //! \brief constructor with position vector initialization
    //! \param[in] pos The position vector which shall be used for initialization
    //----------------------------------------
    TransformationMatrix2d(const Vector2<ValueType>& pos) : m_rotM{RotationMatrix2d<ValueType>{}}, m_posV{pos} {}

    //========================================
    //! \brief constructor with rotation matrix and position vector initialization
    //! \param[in] rm The rotation matrix which shall be used for initialization
    //! \param[in] pos The position vector which shall be used for initialization
    //----------------------------------------
    TransformationMatrix2d(const RotationMatrix2d<ValueType>& rm, const Vector2<ValueType>& pos)
      : m_rotM{rm}, m_posV{pos}
    {}

    //========================================
    //! \brief constructor with angle and position vector initialization
    //! \param[in] angle The angle in [rad] from which the rotation matrix shall
    //! be initialized
    //! \param[in] pos The position vector which shall be used for initialization
    //----------------------------------------
    TransformationMatrix2d(const ValueType angle, const Vector2<ValueType>& pos)
      : m_rotM{RotationMatrix2d<ValueType>{angle}}, m_posV{pos}
    {}

    //========================================
    //! \brief constructor with angle initialization
    //! \param[in] angle The angle in [rad] from which the rotation matrix shall
    //! be initialized
    //----------------------------------------
    TransformationMatrix2d(const ValueType angle)
      : m_rotM{RotationMatrix2d<ValueType>{angle}}, m_posV{Vector2<ValueType>{}}
    {}

public: // operators
    //========================================
    //! \brief Multiplies another Transformation matrix from right hand side to
    //! this one. Performs a transformation of the other matrix into the reference
    //! system of this matrix
    //! \param[in] other The matrix which shall be multiplied to this one
    //! \return A reference to this after the calculation
    //----------------------------------------
    TransformationMatrix2d& operator*=(const TransformationMatrix2d<ValueType>& other)
    {
        m_posV += m_rotM * other.m_posV;
        m_rotM *= other.m_rotM;
        return *this;
    }

    //========================================
    //! \brief Multiplies another Transformation matrix from right hand side to
    //! this one. Performs a transformation of the other matrix into the reference
    //! system of this matrix
    //! \param[in] other The matrix which shall be multiplied to this one
    //! \return A new TransformationMatrix holding the result of the multiplication
    //----------------------------------------
    TransformationMatrix2d operator*(const TransformationMatrix2d<ValueType>& other) const
    {
        return TransformationMatrix2d<ValueType>(*this) *= other;
    }

    //========================================
    //! \brief Performs a transformation of a vector into the reference system of
    //! this matrix. Performs a matrix multiplication T*V
    //! \param[in] vector The vector which shall be transformed
    //! \return The transformed vector
    //----------------------------------------
    Vector2<ValueType> operator*(const Vector2<ValueType>& vector) const { return (m_rotM * vector) + m_posV; }

    //========================================
    //! \brief Checks two matrices for equality
    //! \param[in] other The matrix which shall be compared to this one
    //! \return True, if all elements of both matrices are equal, \c false otherwise
    //----------------------------------------
    bool operator==(const TransformationMatrix2d<ValueType>& other) const
    {
        return m_rotM == other.m_rotM && m_posV == other.m_posV;
    }

    //========================================
    //! \brief Checks two matrices for inequality
    //! \param[in] other The matrix which shall be compared to this one
    //! \return False, if all elements of both matrices are equal, true otherwise
    //----------------------------------------
    bool operator!=(const TransformationMatrix2d<ValueType>& other) const { return !(*this == other); }

public: // member functions
    //========================================
    //!\brief Inverts this transformation matrix. Note that inverting
    //!       transformation matrices is much cheaper than inverting
    //!       normal matrices
    //!\return A reference to this after inverting.
    //----------------------------------------
    TransformationMatrix2d<ValueType>& invert()
    {
        *this = TransformationMatrix2d<ValueType>{*this}.getInverse();
        return *this;
    }

    //========================================
    //! \brief Inverts this transformation matrix. Note that inverting
    //! transformation matrices is much cheaper than inverting normal matrices
    //! \return A new TransformationMatrix which holds the inverse of this matrix
    //----------------------------------------
    TransformationMatrix2d<ValueType> getInverse() const
    {
        RotationMatrix2d<ValueType> invRot{m_rotM.getInverse()};
        Vector2<ValueType> invPos{-invRot * m_posV};
        return TransformationMatrix2d<ValueType>{invRot, invPos};
    }

    //========================================
    //! \brief Rotates the matrix about the x-axis
    //! \param[in] angle The angle in [rad]
    //! \return A reference to this after the rotation
    //----------------------------------------
    TransformationMatrix2d<ValueType>& rotateX(const ValueType angle)
    {
        *this *= TransformationMatrix2d<ValueType>{RotationMatrix2d<ValueType>{}.rotateX(angle)};
        return *this;
    }

    //========================================
    //! \brief Rotates the matrix about the x-axis
    //! \param[in] angle The angle in [rad]
    //! \return A new matrix which is rotated
    //----------------------------------------
    TransformationMatrix2d<ValueType> getRotatedX(const ValueType angle) const
    {
        return TransformationMatrix2d<ValueType>{*this}.rotateX(angle);
    }

    //========================================
    //! \brief Rotates the matrix about the y-axis
    //! \param[in] angle The angle in [rad]
    //! \return A reference to this after the rotation
    //----------------------------------------
    TransformationMatrix2d<ValueType>& rotateY(const ValueType angle)
    {
        *this *= TransformationMatrix2d<ValueType>{RotationMatrix2d<ValueType>{}.rotateY(angle)};
        return *this;
    }

    //========================================
    //! \brief Rotates the matrix about the Y-axis
    //! \param[in] angle The angle in [rad]
    //! \return A new matrix which is rotated
    //----------------------------------------
    TransformationMatrix2d<ValueType> getRotatedY(const ValueType angle) const
    {
        return TransformationMatrix2d<ValueType>{*this}.rotateY(angle);
    }

    //========================================
    //! \brief Translates the matrix about a given offset
    //! \param[in] offset A offset vector
    //! \return A reference to this after translation
    //----------------------------------------
    TransformationMatrix2d<ValueType>& translate(const Vector2<ValueType>& offset)
    {
        *this *= TransformationMatrix2d<ValueType>{offset};
        return *this;
    }

    //========================================
    //! \brief Translates the matrix about a given offset
    //! \param[in] offset A offset vector
    //! \return A new matrix which is translated
    //----------------------------------------
    TransformationMatrix2d<ValueType> getTranslated(const Vector2<ValueType>& offset) const
    {
        return TransformationMatrix2d<ValueType>{*this}.translate(offset);
    }

public: // getters
    //========================================
    //! \brief Getter for the rotation matrix part of the transformation matrix
    //! \return The rotation matrix
    //----------------------------------------
    const RotationMatrix2d<ValueType>& getRotationMatrix() const { return m_rotM; }

    //========================================
    //! \brief Getter for the position vector part of the transformation matrix
    //! \return The position vector
    //----------------------------------------
    const Vector2<ValueType>& getPositionVector() const { return m_posV; }

protected:
    RotationMatrix2d<ValueType> m_rotM;
    Vector2<ValueType> m_posV;

}; // TransformationMatrix2d

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

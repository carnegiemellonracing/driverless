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

#include <microvision/common/sdk/RotationMatrix3d.hpp>
#include <microvision/common/sdk/Vector3.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Transformation matrix class which holds a 3x3 rotation matrix and
//! a 3x1 dimensional position vector. The size of the matrix is 4x4.
//! The structure is:
//!
//! | rotM   posV |
//! | 0 0 0     1 |
//!
//! Dedicated to be used for 3d coordinate transformations
// ------------------------------------------------------------------------------
template<typename T>
class TransformationMatrix3d final
{
public:
    using ValueType = T;

public: // constructors
    //========================================
    //! \brief Default constructors which initializes the matrix to an identity matrix.
    //----------------------------------------
    TransformationMatrix3d() : m_rotM{RotationMatrix3d<ValueType>{}}, m_posV{Vector3<ValueType>{}} {}

    //========================================
    //! \brief Constructor with rotation matrix initialization
    //! \param[in] rotationMatrix  The rotation matrix part of this transformation matrix.
    //----------------------------------------
    TransformationMatrix3d(const RotationMatrix3d<ValueType>& rotationMatrix)
      : m_rotM{rotationMatrix}, m_posV{Vector3<ValueType>{}}
    {}

    //========================================
    //! \brief Constructor with position vector initialization
    //! \param[in] position  The position vector part of this transformation matrix.
    //----------------------------------------
    TransformationMatrix3d(const Vector3<ValueType>& position) : m_rotM{RotationMatrix3d<ValueType>{}}, m_posV{position}
    {}

    //========================================
    //! \brief Constructor with rotation matrix and position vector initialization.
    //! \param[in] rm   The rotation matrix part of this transformation matrix.
    //! \param[in] pos  The position vector part of this transformation matrix.
    //----------------------------------------
    TransformationMatrix3d(const RotationMatrix3d<ValueType>& rotationMatrix, const Vector3<ValueType>& position)
      : m_rotM{rotationMatrix}, m_posV{position}
    {}

public: // operators
    //========================================
    //! \brief Multiplies another transformation matrix from right hand side to
    //!        this one. Performs a transformation of the other matrix into the reference
    //!        system of this matrix
    //! \param[in] other  The matrix to be multiplied from right to this one.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    TransformationMatrix3d& operator*=(const TransformationMatrix3d<ValueType>& other)
    {
        m_posV += m_rotM * other.m_posV;
        m_rotM *= other.m_rotM;
        return *this;
    }

    //========================================
    //! \brief Multiplies another transformation matrix from right hand side to
    //!        this one. Performs a transformation of the other matrix into the reference
    //!        system of this matrix
    //! \param[in] other  The matrix to be multiplied from right to this one.
    //! \return The product of the two matrices.
    //----------------------------------------
    TransformationMatrix3d operator*(const TransformationMatrix3d<ValueType>& other) const
    {
        return TransformationMatrix3d<ValueType>(*this) *= other;
    }

    //========================================
    //! \brief Performs a transformation of a vector into the reference system of
    //!        this matrix and the translation vector t. Performs a matrix multiplication T*v + t.
    //! \param[in] vector  The vector to be translated.
    //! \return The translated vector.
    //----------------------------------------
    Vector3<ValueType> operator*(const Vector3<ValueType>& vector) const { return m_rotM * vector + m_posV; }

    //========================================
    //! \brief Checks two transformation matrices for equality.
    //! \param[in] other  The matrix to be compared to this one.
    //! \return \c True, if the two rotation matrices and the two translation
    //!         vectors are identical. \c false otherwise.
    //----------------------------------------
    bool operator==(const TransformationMatrix3d<ValueType>& other) const
    {
        return (m_rotM == other.m_rotM) && (m_posV == other.m_posV);
    }

    //========================================
    //! \brief Checks two transformation matrices for inequality.
    //! \param[in] other  The matrix to be compared to this one.
    //! \return \c True, if the rotation matrices or the translation vectors of
    //!         the two translation matrices are different. \c false if
    //!         matrices and vectors are identical.
    //----------------------------------------
    bool operator!=(const TransformationMatrix3d<ValueType>& other) const { return !(*this == other); }

public: // member functions
    //========================================
    //! \brief Inverts this transformation matrix.
    //! \return A reference to this .
    //! \note Inverting transformation matrices is much cheaper than inverting normal matrices.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& invert()
    {
        *this = TransformationMatrix3d<ValueType>{*this}.getInverse();
        return *this;
    }

    //========================================
    //! \brief Inverts this transformation matrix.
    //! \return A transformation matrix which is inverse to this.
    //! \note Inverting transformation matrices is much cheaper than inverting normal matrices.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getInverse() const
    {
        RotationMatrix3d<ValueType> invRot{m_rotM.getInverse()};
        Vector3<ValueType> invPos{-invRot * m_posV};
        return TransformationMatrix3d<ValueType>{invRot, invPos};
    }

public:
    //========================================
    //! \brief Rotates the matrix about the x-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundXBefore(const ValueType angle)
    {
        *this *= TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundX(angle)};
        return *this;
    }

    //========================================
    //! \brief Rotates the matrix about the x-axis. Rotation is multiplied from the left side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundXAfter(const ValueType angle)
    {
        *this = TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundX(angle)} * (*this);
        return *this;
    }

public:
    //========================================
    //! \brief Rotates the matrix about the x-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundXBefore(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundXBefore(angle);
    }

    //========================================
    //! \brief Rotates the matrix about the x-axis. Rotation is multiplied from the left side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundXAfter(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundXAfter(angle);
    }

public:
    //========================================
    //! \brief Rotates the matrix about the y-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundYBefore(const ValueType angle)
    {
        *this *= TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundY(angle)};
        return *this;
    }

    //========================================
    //! \brief Rotates the matrix about the y-axis. Rotation is multiplied from the left side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundYAfter(const ValueType angle)
    {
        *this = TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundY(angle)} * (*this);
        return *this;
    }

public:
    //========================================
    //! \brief Rotates the matrix about the Y-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundYBefore(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundYBefore(angle);
    }

    //========================================
    //! \brief Rotates the matrix about the Y-axis. Rotation is multiplied from the left side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundYAfter(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundYAfter(angle);
    }

public:
    //========================================
    //! \brief Rotates the matrix about the z-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundZBefore(const ValueType angle)
    {
        *this *= TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundZ(angle)};
        return *this;
    }

    //========================================
    //! \brief Rotates the matrix about the z-axis. Rotation is multiplied from the left side.
    //! \param[in] angle The counter clockwise rotation angle in [rad].
    //! \return A reference to this after the rotation
    //----------------------------------------
    TransformationMatrix3d<ValueType>& rotateAroundZAfter(const ValueType angle)
    {
        *this = TransformationMatrix3d<ValueType>{RotationMatrix3d<ValueType>{}.setToRotationAroundZ(angle)} * (*this);
        return *this;
    }

public:
    //========================================
    //! \brief Rotates the matrix about the z-axis. Rotation is multiplied from the right side.
    //! \param[in] angle  The rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundZBefore(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundZBefore(angle);
    }

    //========================================
    //! \brief Rotates the matrix about the z-axis. Rotation is multiplied from the left side.
    //! \param[in] angle  The counter clockwise rotation angle in [rad].
    //! \return A new matrix which is rotated.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getRotatedAroundZAfter(const ValueType angle) const
    {
        return TransformationMatrix3d<ValueType>{*this}.rotateAroundZAfter(angle);
    }

public:
    //========================================
    //! \brief Translates the matrix about a given offset. Translation is multiplied from the right side.
    //! \param[in] offset  The translation vector.
    //! \return A reference to this after translation.
    //----------------------------------------
    TransformationMatrix3d<ValueType>& translate(const Vector3<ValueType>& offset)
    {
        *this *= TransformationMatrix3d<ValueType>{offset};
        return *this;
    }

    //========================================
    //! \brief Translates the matrix about a given offset. Translation is multiplied from the right side.
    //! \param[in] offset  The translation vector.
    //! \return A by offset translated transformation matrix.
    //----------------------------------------
    TransformationMatrix3d<ValueType> getTranslated(const Vector3<ValueType>& offset) const
    {
        return TransformationMatrix3d<ValueType>{*this}.translate(offset);
    }

public: // getters
    //========================================
    //! \brief Getter for the rotation matrix part of the transformation matrix
    //! \return The rotation matrix.
    //----------------------------------------
    const RotationMatrix3d<ValueType>& getRotationMatrix() const { return m_rotM; }

    //========================================
    //! \brief Getter for the position vector part of the transformation matrix
    //! \return The position vector.
    //----------------------------------------
    const Vector3<ValueType>& getTranslationVector() const { return m_posV; }

public: // setters
    //========================================
    //! \brief Setter for the rotation matrix part of the transformation matrix.
    //! \param[in] rotation  The new rotation matrix.
    //----------------------------------------
    void setRotationMatrix(const RotationMatrix3d<ValueType>& rotation) { m_rotM = rotation; }

    //========================================
    //! \brief Setter for the position vector part of the transformation matrix.
    //! \param[in] translation  The new position vector.
    //----------------------------------------
    void setTranslationVector(const Vector3<ValueType>& translation) { m_posV = translation; }

protected:
    RotationMatrix3d<ValueType> m_rotM; //!<  The rotation matrix part of the transformation matrix.
    Vector3<ValueType> m_posV; //!< The position vector part of the transformation matrix.

}; // TransformationMatrix3d

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

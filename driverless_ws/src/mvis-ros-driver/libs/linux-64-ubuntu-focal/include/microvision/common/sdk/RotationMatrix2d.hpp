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

#include <microvision/common/sdk/Matrix2x2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Matrix class for which can store a 2x2 rotation matrix.
//!
//! Dedicated to be used for 2d calculations
// ------------------------------------------------------------------------------
template<typename T>
class RotationMatrix2d final : public Matrix2x2<T>
{
public:
    using ValueType = T;

public: // constructors
    //========================================
    //! \brief Default constructors which initializes the matrix to an identity matrix.
    //----------------------------------------
    RotationMatrix2d() : Matrix2x2<ValueType>() {}

    //========================================
    //! \brief Constructor for constructing a rotation matrix rotated about a
    //! given angle
    //! \param[in] alpha The angle in [rad] about the matrix will be rotated
    //----------------------------------------
    RotationMatrix2d(const ValueType alpha)
      : Matrix2x2<ValueType>(std::cos(alpha), -std::sin(alpha), std::sin(alpha), std::cos(alpha))
    {}

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    RotationMatrix2d(const Matrix2x2<ValueType>& matrix) : Matrix2x2<ValueType>(matrix) {}

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    RotationMatrix2d(const MatrixNxN<ValueType, 2>& matrix) : Matrix2x2<ValueType>(matrix) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RotationMatrix2d() override = default;

public: // overloaded functions
    //========================================
    //! \brief Inverts the matrix
    //! \return A reference to this after inverting
    //----------------------------------------
    RotationMatrix2d<ValueType>& invert() final
    {
        *this = getInverse();
        return *this;
    }

    //========================================
    //! \brief Calculates the inverse of the matrix
    //! \return A new matrix holding the result of the calculation
    //----------------------------------------
    RotationMatrix2d<ValueType> getInverse() const { return this->getTransposed(); }

public:
    //========================================
    //! \brief Calculates the rotation angle of the matrix
    //! \return The rotation angle in [rad]
    //----------------------------------------
    ValueType getAngle() const { return std::atan2(this->getValue(1, 0), this->getValue(0, 0)); }

}; // RotationMatrix2d

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

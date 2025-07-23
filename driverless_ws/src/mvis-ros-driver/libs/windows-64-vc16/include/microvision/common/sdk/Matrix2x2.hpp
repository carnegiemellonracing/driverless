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

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/MatrixNxN.hpp>
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
//! \brief Matrix class for which can store a 2x2 matrix.
//!
//! Dedicated to be used for 2d calculations
// ------------------------------------------------------------------------------
template<typename T>
class Matrix2x2 : public MatrixNxN<T, 2>
{
public:
    using MatrixBaseClass = MatrixNxN<T, 2>;
    using ValueType       = typename MatrixBaseClass::ValueType;
    using RowData         = typename MatrixBaseClass::RowData;
    using MatrixData      = typename MatrixBaseClass::MatrixData;

    using MatrixBaseClass::operator*;

public:
    //========================================
    //! \brief Default constructor.
    //!
    //! Initialize the matrix as the identity matrix.
    //----------------------------------------
    Matrix2x2() = default;

    //========================================
    //! \brief Entry wise constructor
    //! \param[in] m00  The matrix element on row 0 column 0.
    //! \param[in] m01  The matrix element on row 0 column 1.
    //! \param[in] m10  The matrix element on row 1 column 0.
    //! \param[in] m11  The matrix element on row 1 column 1.
    //----------------------------------------
    Matrix2x2(const ValueType m00, const ValueType m01, const ValueType m10, const ValueType m11)
      : MatrixBaseClass(MatrixData{RowData{m00, m01}, RowData{m10, m11}})
    {}

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    Matrix2x2(const MatrixBaseClass& matrix) : MatrixBaseClass(matrix) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Matrix2x2() override = default;

public: // member functions
    //========================================
    //! \brief Calculates the determinant of the matrix.
    //! \return The determinant.
    //----------------------------------------
    ValueType getDeterminant() const
    {
        return this->getValue(0, 0) * this->getValue(1, 1) - this->getValue(0, 1) * this->getValue(1, 0);
    }

    //========================================
    //! \brief Replaces the matrix with the adjoint.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    Matrix2x2<ValueType>& adjoint()
    {
        *this = getAdjoint();
        return *this;
    }

    //========================================
    //! \brief Calculates the adjoint of this matrix.
    //! \return A new matrix holding the adjoint of this matrix.
    //----------------------------------------
    Matrix2x2<ValueType> getAdjoint() const
    {
        return Matrix2x2<ValueType>{
            this->getValue(1, 1), -this->getValue(0, 1), -this->getValue(1, 0), this->getValue(0, 0)};
    }

    //========================================
    //! \brief Inverts the matrix
    //! \return A reference to this after inverting
    //----------------------------------------
    virtual Matrix2x2<ValueType>& invert()
    {
        *this = getInverse();
        return *this;
    }

    //========================================
    //! \brief Calculates the inverse of the matrix
    //! \return A new matrix holding the result of the calculation
    //----------------------------------------
    Matrix2x2<ValueType> getInverse() const
    {
        const ValueType det = getDeterminant();
        if (!fuzzyCompareT<7>(det, static_cast<ValueType>(0.0)))
        {
            const ValueType factor = static_cast<ValueType>(1.0) / det;
            return Matrix2x2<ValueType>{this->getValue(1, 1) * factor,
                                        -this->getValue(0, 1) * factor,
                                        -this->getValue(1, 0) * factor,
                                        this->getValue(0, 0) * factor};
        }
        else
        {
            throw std::logic_error("Cannot calculate inverse! A matrix is invertible if and only if the determinant of "
                                   "the matrix is non-zero");
        }
    }

}; // Matrix2x2

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

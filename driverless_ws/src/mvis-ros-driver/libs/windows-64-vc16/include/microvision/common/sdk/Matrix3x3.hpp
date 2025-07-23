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

#include <microvision/common/sdk/io.hpp> //_prototypes.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/MatrixNxN.hpp>

#include <array>
#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Matrix class for which can store a 3x3 matrix.
//!
//! Dedicated to be used for 3d calculations.
// ------------------------------------------------------------------------------
template<typename T>
class Matrix3x3 : public MatrixNxN<T, 3>
{
public:
    using MatrixBaseClass = MatrixNxN<T, 3>;
    using ValueType       = typename MatrixBaseClass::ValueType;
    using RowData         = typename MatrixBaseClass::RowData;
    using MatrixData      = typename MatrixBaseClass::MatrixData;

    static constexpr uint8_t nbOfRowsCols = MatrixBaseClass::nbOfRowsCols;
    static constexpr uint8_t firstRowCol  = MatrixBaseClass::firstRowCol;

    using MatrixBaseClass::operator*;

public:
    //========================================
    //! \brief Constructs the identity matrix. (Default constructor).
    //----------------------------------------
    Matrix3x3() = default;

    //========================================
    //! \brief Entry-wise constructor.
    //! \param[in] m00  The matrix element on row 0 column 0.
    //! \param[in] m01  The matrix element on row 0 column 1.
    //! \param[in] m02  The matrix element on row 0 column 2.
    //! \param[in] m10  The matrix element on row 1 column 0.
    //! \param[in] m11  The matrix element on row 1 column 1.
    //! \param[in] m12  The matrix element on row 1 column 2.
    //! \param[in] m20  The matrix element on row 2 column 0.
    //! \param[in] m21  The matrix element on row 2 column 1.
    //! \param[in] m22  The matrix element on row 2 column 2.
    //----------------------------------------
    Matrix3x3(const ValueType m00,
              const ValueType m01,
              const ValueType m02,
              const ValueType m10,
              const ValueType m11,
              const ValueType m12,
              const ValueType m20,
              const ValueType m21,
              const ValueType m22)
      : MatrixBaseClass(MatrixData{RowData{m00, m01, m02}, RowData{m10, m11, m12}, RowData{m20, m21, m22}})
    {}

    //========================================
    //! \brief Copy constructor taking a matrix of base class type.
    //! \param[in] matrix  The matrix to be copied from.
    //----------------------------------------
    Matrix3x3(const MatrixBaseClass& src) : MatrixBaseClass(src) {}

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Matrix3x3() override = default;

public: // member functions
    //========================================
    //! \brief Get the determinant of the \c this.
    //! \return The determinant of \c this.
    //----------------------------------------
    ValueType getDeterminant() const
    {
        return this->getValue(0, 0)
                   * ((this->getValue(1, 1) * this->getValue(2, 2)) - (this->getValue(1, 2) * this->getValue(2, 1)))
               + this->getValue(0, 1)
                     * ((this->getValue(1, 2) * this->getValue(2, 0)) - (this->getValue(1, 0) * this->getValue(2, 2)))
               + this->getValue(0, 2)
                     * ((this->getValue(1, 0) * this->getValue(2, 1)) - (this->getValue(1, 1) * this->getValue(2, 0)));
    }

    //========================================
    //! \brief Replaces \c this by its adjoint.
    //! \return A reference to \c this, holding the result.
    //! \note Other names for (classical) \b adjoint are \b adjugate and \b adjunct.
    //----------------------------------------
    Matrix3x3<ValueType>& adjoint()
    {
        *this = getAdjoint();
        return *this;
    }

    //========================================
    //! \brief Calculates the adjoint of \c this.
    //! \return A new matrix holding the adjoint of \c this.
    //! \note Other names for (classical) \b adjoint are \b adjugate and \b adjunct.
    //----------------------------------------
    Matrix3x3<ValueType> getAdjoint() const
    {
        return Matrix3x3<ValueType>{static_cast<ValueType>((this->getValue(1, 1) * this->getValue(2, 2))
                                                           - (this->getValue(1, 2) * this->getValue(2, 1))),
                                    static_cast<ValueType>((this->getValue(0, 2) * this->getValue(2, 1))
                                                           - (this->getValue(0, 1) * this->getValue(2, 2))),
                                    static_cast<ValueType>((this->getValue(0, 1) * this->getValue(1, 2))
                                                           - (this->getValue(0, 2) * this->getValue(1, 1))),
                                    static_cast<ValueType>((this->getValue(1, 2) * this->getValue(2, 0))
                                                           - (this->getValue(1, 0) * this->getValue(2, 2))),
                                    static_cast<ValueType>((this->getValue(0, 0) * this->getValue(2, 2))
                                                           - (this->getValue(0, 2) * this->getValue(2, 0))),
                                    static_cast<ValueType>((this->getValue(0, 2) * this->getValue(1, 0))
                                                           - (this->getValue(0, 0) * this->getValue(1, 2))),
                                    static_cast<ValueType>((this->getValue(1, 0) * this->getValue(2, 1))
                                                           - (this->getValue(1, 1) * this->getValue(2, 0))),
                                    static_cast<ValueType>((this->getValue(0, 1) * this->getValue(2, 0))
                                                           - (this->getValue(0, 0) * this->getValue(2, 1))),
                                    static_cast<ValueType>((this->getValue(0, 0) * this->getValue(1, 1))
                                                           - (this->getValue(0, 1) * this->getValue(1, 0)))};
    }

    //========================================
    //! \brief Replaces \c this by its inverse.
    //! \return A reference to \c this, holding the result.
    //! \exception std::logic_error  Will be thrown if there
    //!                              is no inverse of \c this,
    //!                              i.e. if the determinant is 0.
    //----------------------------------------
    virtual Matrix3x3<ValueType>& invert()
    {
        *this = getInverse();
        return *this;
    }

    //========================================
    //! \brief Get the inverse of \c this.
    //! \return A new matrix holding the inverse of \c this.
    //! \exception std::logic_error  Will be thrown if there
    //!                              is no inverse of \c this,
    //!                              i.e. if the determinant is 0.
    //----------------------------------------
    Matrix3x3<ValueType> getInverse() const
    {
        const ValueType det = getDeterminant();
        if ((det < static_cast<ValueType>(0.0)) || (det > static_cast<ValueType>(0.0)))
        {
            return getAdjoint() / det;
        }
        else
        {
            throw std::logic_error("Cannot calculate inverse! A matrix is invertible if and only if the determinant of "
                                   "the matrix is non-zero");
        }
    }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Matrix3x3<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Matrix3x3<TT>& value);

public:
    static constexpr bool isSerializable()
    {
        return ((std::is_integral<ValueType>{} && std::is_signed<ValueType>{}) || std::is_floating_point<ValueType>{});
    }

}; // Matrix3x3

//==============================================================================
// Specializations for serialization.
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Matrix3x3<T>& value)
{
    static_assert(Matrix3x3<T>::isSerializable(), "writeBE is not implemented for given template type of Matrix3x3");

    for (uint8_t row = Matrix3x3<T>::firstRowCol; row < Matrix3x3<T>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = Matrix3x3<T>::firstRowCol; col < Matrix3x3<T>::nbOfRowsCols; ++col)
        {
            microvision::common::sdk::writeBE(os, value.getValue(row, col));
        }
    }
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, Matrix3x3<T>& value)
{
    static_assert(Matrix3x3<T>::isSerializable(), "readBE is not implemented for given template type of Matrix3x3");

    for (uint8_t row = Matrix3x3<T>::firstRowCol; row < Matrix3x3<T>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = Matrix3x3<T>::firstRowCol; col < Matrix3x3<T>::nbOfRowsCols; ++col)
        {
            T element = static_cast<T>(0);
            microvision::common::sdk::readBE(is, element);
            value.setValue(row, col, element);
        }
    }
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const Matrix3x3<T>&)
{
    static_assert(Matrix3x3<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of Matrix3x3");
    return Matrix3x3<T>::nbOfRowsCols * Matrix3x3<T>::nbOfRowsCols * sizeof(T);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

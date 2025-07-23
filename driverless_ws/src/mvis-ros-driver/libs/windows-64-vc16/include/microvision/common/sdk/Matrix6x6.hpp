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

#include <microvision/common/sdk/io.hpp> //_prototypes.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/MatrixNxN.hpp>

#include <iostream>
#include <array>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Matrix class for which can store a 6x6 matrix.
//!
//! Dedicated to be used as storage or for simple 6d calculations
// ------------------------------------------------------------------------------
template<typename T>
class Matrix6x6 : public MatrixNxN<T, 6>
{
public:
    using MatrixBaseClass = MatrixNxN<T, 6>;
    using ValueType       = typename MatrixBaseClass::ValueType;
    using RowData         = typename MatrixBaseClass::RowData;
    using MatrixData      = typename MatrixBaseClass::MatrixData;

public:
    //========================================
    //! \brief Default constructor.
    //!
    //! Initialize the matrix as the identity matrix.
    //----------------------------------------
    Matrix6x6() = default;

    //========================================
    //! \brief Entry wise constructor.
    //----------------------------------------
    Matrix6x6(const ValueType m00,
              const ValueType m01,
              const ValueType m02,
              const ValueType m03,
              const ValueType m04,
              const ValueType m05,

              const ValueType m10,
              const ValueType m11,
              const ValueType m12,
              const ValueType m13,
              const ValueType m14,
              const ValueType m15,

              const ValueType m20,
              const ValueType m21,
              const ValueType m22,
              const ValueType m23,
              const ValueType m24,
              const ValueType m25,

              const ValueType m30,
              const ValueType m31,
              const ValueType m32,
              const ValueType m33,
              const ValueType m34,
              const ValueType m35,

              const ValueType m40,
              const ValueType m41,
              const ValueType m42,
              const ValueType m43,
              const ValueType m44,
              const ValueType m45,

              const ValueType m50,
              const ValueType m51,
              const ValueType m52,
              const ValueType m53,
              const ValueType m54,
              const ValueType m55)
      : MatrixBaseClass(MatrixData{RowData{m00, m01, m02, m03, m04, m05},
                                   RowData{m10, m11, m12, m13, m14, m15},
                                   RowData{m20, m21, m22, m23, m24, m25},
                                   RowData{m30, m31, m32, m33, m34, m35},
                                   RowData{m40, m41, m42, m43, m44, m45},
                                   RowData{m50, m51, m52, m53, m54, m55}})
    {}

    virtual ~Matrix6x6() = default;

public: // getter functions
    //========================================
    //! \brief Getter for partitioned matrix into four 3×3 blocks
    //! \return The block matrix11 as Matrix3x3
    MatrixNxN<ValueType, 3> get3x3BlockMatrix00() const
    {
        return MatrixNxN<ValueType, 3>{this->getValue(0, 0),
                                       this->getValue(0, 1),
                                       this->getValue(0, 2),
                                       this->getValue(1, 0),
                                       this->getValue(1, 1),
                                       this->getValue(1, 2),
                                       this->getValue(2, 0),
                                       this->getValue(2, 1),
                                       this->getValue(2, 2)};
    }

    //========================================
    //! \brief Getter for partitioned matrix into four 3×3 blocks
    //! \return The block matrix12 as Matrix3x3
    MatrixNxN<ValueType, 3> get3x3BlockMatrix01() const
    {
        return MatrixNxN<ValueType, 3>{this->getValue(0, 3),
                                       this->getValue(0, 4),
                                       this->getValue(0, 5),
                                       this->getValue(1, 3),
                                       this->getValue(1, 4),
                                       this->getValue(1, 5),
                                       this->getValue(2, 3),
                                       this->getValue(2, 4),
                                       this->getValue(2, 5)};
    }

    //========================================
    //! \brief Getter for partitioned matrix into four 3×3 blocks
    //! \return The block matrix21 as Matrix3x3
    MatrixNxN<ValueType, 3> get3x3BlockMatrix10() const
    {
        return MatrixNxN<ValueType, 3>{this->getValue(3, 0),
                                       this->getValue(3, 1),
                                       this->getValue(3, 2),
                                       this->getValue(4, 0),
                                       this->getValue(4, 1),
                                       this->getValue(4, 2),
                                       this->getValue(5, 0),
                                       this->getValue(5, 1),
                                       this->getValue(5, 2)};
    }

    //========================================
    //! \brief Getter for partitioned matrix into four 3×3 blocks
    //! \return The block matrix22 as Matrix3x3
    MatrixNxN<ValueType, 3> get3x3BlockMatrix11() const
    {
        return MatrixNxN<ValueType, 3>{this->getValue(3, 3),
                                       this->getValue(3, 4),
                                       this->getValue(3, 5),
                                       this->getValue(4, 3),
                                       this->getValue(4, 4),
                                       this->getValue(4, 5),
                                       this->getValue(5, 3),
                                       this->getValue(5, 4),
                                       this->getValue(5, 5)};
    }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, Matrix6x6<TT>& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const Matrix6x6<TT>& value);

public:
    static constexpr bool isSerializable()
    {
        return ((std::is_integral<ValueType>{} && std::is_signed<ValueType>{}) || std::is_floating_point<ValueType>{});
    }
}; // Matrix6x6

//==============================================================================
// Specializations for serialization.
//==============================================================================

template<typename T>
inline void writeBE(std::ostream& os, const Matrix6x6<T>& value)
{
    static_assert(Matrix6x6<T>::isSerializable(), "writeBE is not implemented for given template type of Matrix6x6");

    for (uint8_t i = 0; i < 6; ++i)
    {
        for (uint8_t j = 0; j < 6; ++j)
        {
            microvision::common::sdk::writeBE(os, value.getValue(i, j));
        }
    }
}

//==============================================================================

template<typename T>
inline void readBE(std::istream& is, Matrix6x6<T>& value)
{
    static_assert(Matrix6x6<T>::isSerializable(), "readBE is not implemented for given template type of Matrix6x6");

    for (uint8_t i = 0; i < 6; ++i)
    {
        for (uint8_t j = 0; j < 6; ++j)
        {
            T element = 0;
            microvision::common::sdk::readBE(is, element);
            value.setValue(i, j, element);
        }
    }
}

//==============================================================================

template<typename T>
inline constexpr std::streamsize serializedSize(const Matrix6x6<T>&)
{
    static_assert(Matrix6x6<T>::isSerializable(),
                  "serializedSize is not implemented for given template type of Matrix6x6");
    return 36 * sizeof(T);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
//! \date Dec 18, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io_prototypes.hpp>
#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/Vector3.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <array>
#include <iostream>
#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Matrix class for which can store a nxN matrix.
//!
//! Dedicated to be used for nd calculations.
// ------------------------------------------------------------------------------
template<typename T, uint8_t n>
class MatrixNxN
{
public:
    using ValueType = T;

    static constexpr uint8_t nbOfRowsCols{n};
    static constexpr uint8_t firstRowCol{0U};

    using RowData    = std::array<ValueType, nbOfRowsCols>;
    using MatrixData = std::array<RowData, nbOfRowsCols>;

public:
    //========================================
    //! \brief Constructs an identity matrix. (Default constructor).
    //----------------------------------------
    MatrixNxN() : m_elements{} { setToIdentity(); }

    //========================================
    //! \brief Constructor.
    //----------------------------------------
    explicit MatrixNxN(const MatrixData& matrix) : m_elements{matrix} {}

    //========================================
    //! \brief Default copy constructor.
    //----------------------------------------
    MatrixNxN(const MatrixNxN& matrix) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~MatrixNxN() = default;

public: // operators
    //========================================
    //! \brief Performs an in-place entry-wise addition of two matrices M1 + M2, with M1 = \c this.
    //! \param[in] other  The matrix to be added to \c this.
    //! \return A reference to \c this, holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& operator+=(const MatrixNxN<ValueType, n>& other)
    {
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < nbOfRowsCols; ++col)
            {
                m_elements[row][col] += other.getValue(row, col);
            }
        }
        return *this;
    }

    //========================================
    //! \brief Performs an entry-wise addition of two matrices M1 + M2, with M1 = \c this.
    //! \param[in] other  The matrix to be added to \c this one.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator+(const MatrixNxN<ValueType, n>& other) const
    {
        return MatrixNxN<ValueType, n>{*this} += other;
    }

    //========================================
    //! \brief Performs an in-place entry-wise subtraction of two matrices M1 - M2, with M1 = \c this.
    //! \param[in] other  The matrix to be subtracted from \c this one.
    //! \return A reference to \c this, holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& operator-=(const MatrixNxN<ValueType, n>& other)
    {
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < nbOfRowsCols; ++col)
            {
                m_elements[row][col] -= other.getValue(row, col);
            }
        }
        return *this;
    }

    //========================================
    //! \brief Performs an entry-wise subtraction of two matrices M1 - M2, with M1 = \c this.
    //! \param[in] other  The matrix to be subtracted from \c this.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator-(const MatrixNxN<ValueType, n>& other) const
    {
        return MatrixNxN<ValueType, n>{*this} -= other;
    }

    //========================================
    //! \brief Performs an in-place entry-wise multiplication with a \a scalar.
    //! \param[in] scalar  the factor to be used for the calculation.
    //! \return A reference to \c this, holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& operator*=(const ValueType scalar)
    {
        for (auto& row : m_elements)
        {
            for (auto& element : row) // for all columns
            {
                element *= scalar;
            }
        }
        return *this;
    }

    //========================================
    //! \brief Performs an entry-wise multiplication with a \a scalar.
    //! \param[in] scalar  The scalar factor the matrix to be multiplied with.
    //! \return A matrix as the result of the multiplication M * s, with M = \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator*(const ValueType scalar) const { return MatrixNxN<ValueType, n>{*this} *= scalar; }

    //========================================
    //! \brief Performs an entry-wise multiplication of a \a matrix with a \a scalar.
    //! \param[in] scalar  The scalar factor the matrix to be multiplied with entry-wise.
    //! \param[in] matrix  The matrix to be multiplied by the \a scalar.
    //! \return A new Matrix holding the result of scalar * matrix.
    //----------------------------------------
    friend MatrixNxN<ValueType, n> operator*(const ValueType scalar, const MatrixNxN<ValueType, n>& matrix)
    {
        // multiplication with a scalar is commutative
        return MatrixNxN<ValueType, n>{matrix * scalar};
    }

    //========================================
    //! \brief Performs an in-place matrix-matrix multiplication: M1 * M2, with M1 = \c this.
    //! \param[in] other  The matrix to be multiplied from the right to \c this.
    //! \return A reference to \c this, holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& operator*=(const MatrixNxN<ValueType, n>& other)
    {
        MatrixNxN<ValueType, n> product;
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < nbOfRowsCols; ++col)
            {
                ValueType m = 0;
                for (uint8_t sum = firstRowCol; sum < nbOfRowsCols; ++sum)
                {
                    m += m_elements[row][sum] * other.m_elements[sum][col];
                }
                product.setValue(row, col, m);
            }
        }
        *this = product;
        return *this;
    }

    //========================================
    //! \brief Performs a matrix-matrix multiplication M1 * M2 with M1 = \c this.
    //! \param[in] other  The matrix to be multiplied from the right to \c this.
    //! \return A new matrix holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator*(const MatrixNxN<ValueType, n>& other) const
    {
        return MatrixNxN<ValueType, n>{*this} *= other;
    }

    //========================================
    //! \brief Performs an in-place entry-wise division of \c this by a divisor.
    //! \param[in] divisor  The divisor for the division calculation.
    //! \return A reference to \c this, holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& operator/=(const ValueType divisor)
    {
        const ValueType rezDivisor = static_cast<ValueType>(1.0) / divisor;
        return operator*=(rezDivisor);
    }

    //========================================
    //! \brief Performs an entry-wise division of \c this by a divisor.
    //! \param[in] division  The divisor for the division calculation.
    //! \return A new matrix holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator/(const ValueType divisor) const
    {
        return MatrixNxN<ValueType, n>{*this} /= divisor;
    }

    //========================================
    //! \brief negates the \c this.
    //! \return The negative matrix of \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n> operator-() const
    {
        MatrixNxN<ValueType, n> matrix;
        matrix.setToZero();
        matrix -= *this;
        return matrix;
    }

public: // member functions
    //========================================
    //! \brief Transposes \c this in-place.
    //! \return A reference to \c this holding the result.
    //----------------------------------------
    MatrixNxN<ValueType, n>& transpose()
    {
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < row; ++col)
            {
                // Swapping row and column of the non-diagonal entries.
                std::swap(m_elements[row][col], m_elements[col][row]);
            }
        }
        return *this;
    }

    //========================================
    //! \brief Get the transposed matrix of \c this.
    //! \return A new matrix which is the transposed of \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n> getTransposed() const { return MatrixNxN<ValueType, n>{*this}.transpose(); }

    //========================================
    //! \brief Resets \c this matrix to identity.
    //! \return A reference to \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n>& setToIdentity()
    {
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < nbOfRowsCols; ++col)
            {
                if (row == col)
                {
                    m_elements[row][col] = 1;
                }
                else
                {
                    m_elements[row][col] = 0;
                }
            }
        }
        return *this;
    }

    //========================================
    //! \brief Resets \c this matrix to zero.
    //! \return A reference to \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n>& setToZero() { return this->fill(0); }

    //========================================
    //! \brief Set whole matrix to \b defaultValue.
    //! \param[in] defaultValue  The value which has to be set.
    //! \return A reference to \c this.
    //----------------------------------------
    MatrixNxN<ValueType, n>& fill(const ValueType defaultValue)
    {
        for (uint8_t row = firstRowCol; row < nbOfRowsCols; ++row)
        {
            for (uint8_t col = firstRowCol; col < nbOfRowsCols; ++col)
            {
                m_elements[row][col] = defaultValue;
            }
        }
        return *this;
    }

    // TODO
    // MatrixNxN<FloatingType, n>& adjoint() {}
    // MatrixNxN<FloatingType, n> getAdjoint() const {}
    // MatrixNxN<FloatingType, n>& invert() {}
    // MatrixNxN<FloatingType, n> getInverse() const {}
    // FloatingType getDeterminant() const {}

public: // getter functions
    // ========================================
    //! \brief Gets the value of the matrix element for given \a row and \a column.
    //! \param[in] row  The row of the requested element. Counting starts at 0.
    //! \param[in] col  The column of the requested element. Counting starts at 0.
    //! \return The matrix element value at the given \a row and \a column.
    //! \attention There is no bounds checking for \a row and \a column. To perform
    //!            a bounds checking use getValueChecked() instead.
    //! \sa getValueChecked()
    //----------------------------------------
    ValueType getValue(const uint8_t row, const uint8_t col) const { return m_elements[row][col]; }

    // ========================================
    //! \brief Gets the value of the matrix element for given \a row and \a column.
    //! \param[in] row  The row of the requested element. Counting starts at 0.
    //! \param[in] col  The column of the requested element. Counting starts at 0.
    //! \return The matrix element value at the given \a row and \a column.
    //! \attention Bounds checking for \a row and \a column will be performed.
    //!            If you want to avoid this due to performance reasons, use getValue()
    //!            instead.
    //! \sa getValue()
    //----------------------------------------
    ValueType getValueChecked(const uint8_t row, const uint8_t col) const { return m_elements.at(row).at(col); }

    // ========================================
    //! \brief Sets the value of the matrix element for given \a row and \a column.
    //! \param[in] row  The row of the requested element. Counting starts at 0.
    //! \param[in] col  The column of the requested element. Counting starts at 0.
    //! \param[in] value  The new value for the matrix entry at \a row and \a column.
    //! \attention There is no bounds checking for \a row and \a column. To perform
    //!            a bounds checking use setValueChecked() instead.
    //! \sa setValueChecked()
    //----------------------------------------
    void setValue(const uint8_t row, const uint8_t col, const ValueType value) { m_elements[row][col] = value; }

    // ========================================
    //! \brief Sets the value of the matrix element for given \a row and \a column.
    //! \param[in] row  The row of the requested element. Counting starts at 0.
    //! \param[in] col  The column of the requested element. Counting starts at 0.
    //! \param[in] value  The new value for the matrix entry at \a row and \a column.
    //! \attention Bounds checking for \a row and \a column will be performed.
    //!            If you want to avoid this due to performance reasons, use setValue()
    //!            instead.
    //! \sa setValue()
    //----------------------------------------
    void setValueChecked(const uint8_t row, const uint8_t col, const ValueType value)
    {
        m_elements.at(row).at(col) = value;
    }

    // ========================================
    //! \brief Gets the row array of the matrix for given \a row.
    //! \param[in] row  The row number of the requested row. Counting starts at 0.
    //! \return The row array at the given \a row.
    //! \attention There is no bounds checking for \a row. To perform
    //!            a bounds checking use getRowChecked() instead.
    //! \sa getRowChecked()
    //----------------------------------------
    const RowData& getRow(const uint8_t row) const { return m_elements[row]; }

    // ========================================
    //! \brief Gets the row array of the matrix for given \a row.
    //! \param[in] row  The row number of the requested row. Counting starts at 0.
    //! \return The row array at the given \a row.
    //! \attention Bounds checking for \a row will be performed.
    //!            If you want to avoid this due to performance reasons, use getRow()
    //!            instead.
    //! \sa getRow()
    //----------------------------------------
    const RowData& getRowChecked(const uint8_t row) const { return m_elements.at(row); }

    // ========================================
    //! \brief Gets the array of the matrix elements.
    //! \return The array a reference to the matrix elements.
    //----------------------------------------
    const MatrixData& getMatrix() const { return m_elements; }

protected:
    MatrixData m_elements; //!< array of array with n times n elements.
}; // MatrixNxN

//==============================================================================
// Specializations for operators.
//==============================================================================

//==============================================================================
//! \brief Performs an entry-wise equality check.
//! \param[in] lhs  The first matrix to be compared.
//! \param[in] rhs  The second matrix to be compared.
//! \return \c True, if \c lhs and \c rhs are identically, \c false otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t n>
inline bool operator==(const MatrixNxN<T, n>& lhs, const MatrixNxN<T, n>& rhs)
{
    for (uint8_t row = MatrixNxN<T, n>::firstRowCol; row < MatrixNxN<T, n>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = MatrixNxN<T, n>::firstRowCol; col < MatrixNxN<T, n>::nbOfRowsCols; ++col)
        {
            if (!(lhs.getValue(row, col) == rhs.getValue(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise check, whether \c lhs and the \a rhs matrix are not identically.
//! \param[in] lhs  The first matrix to be compared.
//! \param[in] rhs  The second matrix to be compared.
//! \return \c false if both matrices are identically, \c true otherwise.
//------------------------------------------------------------------------------
template<typename T, uint8_t n>
inline bool operator!=(const MatrixNxN<T, n>& lhs, const MatrixNxN<T, n>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
//! \brief Performs an entry-wise fuzzy equality check (special implementation for float).
//! \tparam EXP     The exponent of the epsilon used for fuzzy compare. 10^(-EXP).
//! \param[in] lhs  The matrix to be compared to.
//! \param[in] rhs  The matrix to be compared with.
//! \return \c true, if for each element of both matrices the difference is not smaller than 10^(-EXP) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n, uint8_t EXP>
bool fuzzyFloatEqualT(const MatrixNxN<float, n>& lhs, const MatrixNxN<float, n>& rhs)
{
    for (uint8_t row = MatrixNxN<float, n>::firstRowCol; row < MatrixNxN<float, n>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = MatrixNxN<float, n>::firstRowCol; col < MatrixNxN<float, n>::nbOfRowsCols; ++col)
        {
            if (fuzzyFloatUnequalT<EXP>(lhs.getValue(row, col), rhs.getValue(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise fuzzy equality check (special implementation for double).
//! \tparam EXP     The exponent of the epsilon used for fuzzy compare. 10^(-EXP).
//! \param[in] lhs  The matrix to be compared to.
//! \param[in] rhs  The matrix to be compared with.
//! \return \c true, if for each element of both matrices the difference is not smaller than 10^(-EXP) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n, uint8_t EXP>
bool fuzzyDoubleEqualT(const MatrixNxN<double, n>& lhs, const MatrixNxN<double, n>& rhs)
{
    for (uint8_t row = MatrixNxN<double, n>::firstRowCol; row < MatrixNxN<double, n>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = MatrixNxN<double, n>::firstRowCol; col < MatrixNxN<double, n>::nbOfRowsCols; ++col)
        {
            if (fuzzyDoubleUnequalT<EXP>(lhs.getValue(row, col), rhs.getValue(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise equality check (special implementation for float).
//! \param[in] lhs  The matrix to be compared to.
//! \param[in] rhs  The matrix to be compared with.
//! \return \c True, if for each element of both matrices the difference is not smaller than 10^(-12) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n>
inline bool operator==(const MatrixNxN<float, n>& lhs, const MatrixNxN<float, n>& rhs)
{
    for (uint8_t row = MatrixNxN<float, n>::firstRowCol; row < MatrixNxN<float, n>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = MatrixNxN<float, n>::firstRowCol; col < MatrixNxN<float, n>::nbOfRowsCols; ++col)
        {
            if (fuzzyFloatUnequalT<12>(lhs.getValue(row, col), rhs.getValue(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs an entry-wise equality check (special implementation for double).
//! \param[in] lhs  The matrix to be compared to.
//! \param[in] rhs  The matrix to be compared with.
//! \return \c True, if for each element of both matrices the difference is not smaller than 10^(-17) or both are NaN,
//!         \c false otherwise.
//------------------------------------------------------------------------------
template<uint8_t n>
inline bool operator==(const MatrixNxN<double, n>& lhs, const MatrixNxN<double, n>& rhs)
{
    for (uint8_t row = MatrixNxN<double, n>::firstRowCol; row < MatrixNxN<double, n>::nbOfRowsCols; ++row)
    {
        for (uint8_t col = MatrixNxN<double, n>::firstRowCol; col < MatrixNxN<double, n>::nbOfRowsCols; ++col)
        {
            if (fuzzyDoubleUnequalT<17>(lhs.getValue(row, col), rhs.getValue(row, col)))
            {
                return false;
            }
        }
    }
    return true;
}

//==============================================================================
//! \brief Performs a matrix-vector multiplication M*v.
//! \param[in] matrix  The matrix the vector is multiplied to.
//! \param[in] vector  The vector multiplied to the matrix from the right.
//! \return A vector holding the result of the calculation M*v.
//------------------------------------------------------------------------------
template<typename T, uint8_t n>
VectorN<T, n> operator*(const MatrixNxN<T, n>& matrix, const VectorN<T, n>& vector)
{
    VectorN<T, n> result;

    for (uint8_t row = MatrixNxN<T, n>::firstRowCol; row < MatrixNxN<T, n>::nbOfRowsCols; ++row)
    {
        T m = 0;
        for (uint8_t col = MatrixNxN<T, n>::firstRowCol; col < MatrixNxN<T, n>::nbOfRowsCols; ++col)
        {
            m += matrix.getValue(row, col) * vector.getValue(col);
        }
        result.setValue(row, m);
    }

    return result;
}

//==============================================================================
//! \brief Performs a vector-matrix multiplication v*M.
//! \param[in] vector  The vector to be multiplied to the matrix from the left.
//! \param[in] matrix  The matrix the vector is multiplied to.
//! \return A vector holding the result of the calculation v*M.
//------------------------------------------------------------------------------
template<typename T, uint8_t n>
VectorN<T, n> operator*(const VectorN<T, n>& vector, const MatrixNxN<T, n>& matrix)
{
    VectorN<T, n> result;
    for (uint8_t col = MatrixNxN<T, n>::firstRowCol; col < MatrixNxN<T, n>::nbOfRowsCols; ++col)
    {
        T m = 0;
        for (uint8_t row = MatrixNxN<T, n>::firstRowCol; row < MatrixNxN<T, n>::nbOfRowsCols; ++row)
        {
            m += vector.getValue(row) * matrix.getValue(row, col);
        }
        result.setValue(col, m);
    }
    return result;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

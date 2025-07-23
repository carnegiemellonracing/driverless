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
//! \date Seb 25, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>

#include <iostream>
#include <array>
#include <limits>
#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Matrix class for which can store a matrix any size.
//!
//! Dedicated to be used as storage or for simple calculations.
//! ToDo: It is not an efficient implementation jet!
// ------------------------------------------------------------------------------
template<typename T>
class Matrix
{
public:
    using ValueType = T;

public:
    //========================================
    //! \brief Standard constructor for null matrix.
    //! \param[in] rows  The number of rows in this matrix.
    //! \param[in] cols  The number of cols in this matrix.
    //----------------------------------------
    Matrix(const uint16_t rows, const uint16_t cols) : m_nbOfRows{rows}, m_nbOfCols{cols}
    {
        m_matrix.resize(rows);
        for (std::vector<ValueType>& vec : m_matrix)
        {
            vec.resize(cols, 0);
        }
    }

    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    Matrix() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~Matrix() = default;

public: // operators
    //========================================
    //! \brief Adds another matrix to this one.
    //! \param[in] other  The matrix which shall be added to this one.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    Matrix<ValueType>& operator+=(const Matrix<ValueType>& other)
    {
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; j < m_nbOfCols; ++j)
            {
                m_matrix[i][j] += other.getValue(i, j);
            }
        }

        return *this;
    }

    //========================================
    //! \brief Adds another matrix to this one.
    //! \param[in] other  The matrix which shall be added to this one.
    //! \return A new matrix holding the result of the calculation.
    Matrix<ValueType> operator+(const Matrix<ValueType>& other) const { return Matrix<ValueType>{*this} += other; }

    //========================================
    //! \brief Subtracts another matrix from this one.
    //! \param[in] other  The matrix which shall be subtracted from this one.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    Matrix<ValueType>& operator-=(const Matrix<ValueType>& other)
    {
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                m_matrix[i][j] -= other.getValue(i, j);
            }
        }
        return *this;
    }

    //========================================
    //! \brief Subtracts another matrix from this one.
    //! \param[in] other  The matrix which shall be subtracted from this one.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    Matrix<ValueType> operator-(const Matrix<ValueType>& other) const { return Matrix<ValueType>{*this} -= other; }

    //========================================
    //! \brief Multiplies the matrix by a factor.
    //! \param[in] factor  The factor which shall be used for the calculation.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    Matrix<ValueType>& operator*=(const ValueType factor)
    {
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                m_matrix[i][j] *= factor;
            }
        }
        return *this;
    }

    //========================================
    //! \brief Multiplies the matrix by a factor.
    //! \param[in] factor  The factor which shall be used for the calculation.
    //! \return A new Matrix holding the result of the calculation.
    //----------------------------------------
    Matrix<ValueType> operator*(const ValueType factor) const { return Matrix<ValueType>{*this} *= factor; }

    //========================================
    //! \brief Matrix-Matrix multiplication M*M.
    //! \param[in] other  The matrix which shall by multiplied to this one.
    //! \return A reference to this after the calculation.
    //----------------------------------------
    Matrix<ValueType>& operator*=(const Matrix<ValueType>& other)
    {
        Matrix<ValueType> mat(m_nbOfRows, m_nbOfCols);
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                ValueType m = 0;
                for (uint16_t k = 0; k < 6; ++k)
                {
                    m += m_matrix[i][k] * other.getValue(k, j);
                }
                mat.setValue(i, j, m);
            }
        }

        *this = mat;
        return *this;
    }

    //========================================
    //! \brief Matrix-Matrix multiplication M*M.
    //! \param[in] other  The matrix which shall by multiplied to this one.
    //! \return The result of the calculation.
    //----------------------------------------
    Matrix<ValueType> operator*(const Matrix<ValueType>& other) const { return Matrix<ValueType>{*this} *= other; }

    //========================================
    //! \brief Matrix division by a divisor. Divides all matrix entries by a given divisor.
    //! \param[in] division  The divisor for the division calculation.
    //! \return A reference to this after the calculation.
    //! \note Beware of rounding Errors dependent of the ValueType.
    //----------------------------------------
    Matrix<ValueType>& operator/=(const ValueType divisor)
    {
        const ValueType rezDivisor = static_cast<ValueType>(1.0) / divisor;
        return operator*=(rezDivisor);
    }

    //========================================
    //! \brief Matrix division by a divisor. Divides all matrix entries by a given divisor.
    //! \param[in] division  The divisor for the division calculation.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    Matrix<ValueType> operator/(const ValueType divisor) const { return Matrix<ValueType>{*this} /= divisor; }

    //========================================
    //! \brief Negates the matrix.
    //! \return A new matrix which is negated.
    //----------------------------------------
    Matrix<ValueType> operator-() const { return Matrix<ValueType>{m_nbOfRows, m_nbOfCols} -= *this; }

    //========================================
    //! \brief Checks to matrices for equality.
    //! \param[in] other  The matrix, this matrix shall be compared to.
    //! \return True, if all elements in the matrix are identical to the entries of the other matrix, false otherwise.
    //! \note Beware of rounding Errors dependent of the ValueType.
    //----------------------------------------
    bool operator==(const Matrix<ValueType>& other) const
    {
        bool ret = true;
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                ret = ret
                      && ((std::isnan(m_matrix[i][j]) && std::isnan(other.getValue(i, j)))
                          || (m_matrix[i][j] == other.getValue(i, j)));
            }
        }
        return ret;
    }

    //========================================
    //! \brief Checks to matrices for inequality.
    //! \param[in] other  The matrix, this matrix shall be compared to.
    //! \return False, if all elements in the matrix are identical to the entries of the other matrix, true otherwise.
    //! \note Beware of rounding Errors dependent of the ValueType.
    //----------------------------------------
    bool operator!=(const Matrix<ValueType>& other) const { return !(*this == other); }

    //TODO add == and != operators for double and float

public: // member functions
    //========================================
    //! \brief Transposes the matrix.
    //! \return A reference to this after transposing.
    //----------------------------------------
    Matrix<ValueType>& transpose()
    {
        *this = getTransposed();
        return *this;
    }

    //========================================
    //! \brief Calculates the transposed of the matrix.
    //! \return A new matrix holding the result of the calculation.
    //----------------------------------------
    Matrix<ValueType> getTransposed() const
    {
        Matrix<ValueType> mat;
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                mat.setValue(i, j, m_matrix[j][i]);
            }
        }
        return mat;
    }

    //========================================
    //! \brief Resets this matrix to hold an identity matrix.
    //! \return A reference to this after resetting to identity.
    //----------------------------------------
    Matrix<ValueType>& setToIdentity()
    {
        for (uint16_t i = 0; i < m_nbOfRows; ++i)
        {
            for (uint16_t j = 0; i < m_nbOfCols; ++j)
            {
                if (i == j)
                {
                    m_matrix[i][j] = 1;
                }
                else
                {
                    m_matrix[i][j] = 0;
                }
            }
        }
        return *this;
    }

public: // getter functions
    //========================================
    //! \brief Get the number of rows in this matrix.
    //! \return The number of rows.
    //----------------------------------------
    uint16_t getRows() const { return m_nbOfRows; }

    //========================================
    //! \brief Get the number of columns in this matrix.
    //! \return The number of columns.
    //----------------------------------------
    uint16_t getCols() const { return m_nbOfCols; }

    //========================================
    //! \brief Get the matrix as vector of vectors.
    //! \return The actual matrix.
    //----------------------------------------
    std::vector<std::vector<ValueType>> getMatrix() { return m_matrix; }

    //========================================
    //! \brief Getter for element wise access to the elements.
    //! \param[in] row  The row index of the element to access. Starts at 0.
    //! \param[in] col  The row index of the element to access. Starts at 0.
    //! \return The matrix element value at the given indices.
    //----------------------------------------
    ValueType getValue(const uint16_t row, const uint16_t col) const { return m_matrix[row][col]; }

    //========================================
    //! \brief Getter for element wise access to the elements with checked boundary.
    //! \param[in] row  The row index of the element to access. Starts at 0.
    //! \param[in] col  The row index of the element to access. Starts at 0.
    //! \return The matrix element value at the given indices.
    //----------------------------------------
    ValueType getValueChecked(const uint16_t row, const uint16_t col) const { return m_matrix.at(row).at(col); }

    //========================================
    //! \brief Setter for element wise access to the elements.
    //! \param[in] row  The row index of the element to access. Starts at 0.
    //! \param[in] col  The row index of the element to access. Starts at 0.
    //! \param[in] value The new value of the element.
    //----------------------------------------
    void setValue(const uint16_t row, const uint16_t col, const ValueType value) { m_matrix[row][col] = value; }

    //========================================
    //! \brief Setter for element wise access to the elements with checked boundary.
    //! \param[in] row  The row index of the element to access. Starts at 0.
    //! \param[in] col  The row index of the element to access. Starts at 0.
    //! \param[in] value The new value of the element.
    //----------------------------------------
    void setValueChecked(const uint16_t row, const uint16_t col, const ValueType value)
    {
        m_matrix.at(row).at(col) = value;
    }

protected:
    std::vector<std::vector<ValueType>> m_matrix; //!< The matrix as vector of vectors.
    uint16_t m_nbOfRows; //!< The number of rows of the matrix.
    uint16_t m_nbOfCols; //!< The number of cols of the matrix.
}; // Matrix

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
//! \date Nov 4, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class SerialNumberIn6301 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    SerialNumberIn6301()          = default;
    virtual ~SerialNumberIn6301() = default;

public:
    //========================================
    //! \brief Get the month.
    //! \return The month.
    //----------------------------------------
    uint8_t getMonth() const { return m_month; }

    //========================================
    //! \brief Get the year.
    //! \return The year.
    //----------------------------------------
    uint8_t getYear() const { return m_year; }

    //========================================
    //! \brief Get the cnt1.
    //! \return The cnt1.
    //----------------------------------------
    uint8_t getCnt1() const { return m_cnt1; }

    //========================================
    //! \brief Get the cnt0.
    //! \return The cnt0.
    //----------------------------------------
    uint8_t getCnt0() const { return m_cnt0; }

    //========================================
    //! \brief Get the null.
    //! \return The null.
    //----------------------------------------
    uint16_t getNull() const { return m_null; }

public:
    //========================================
    //! \brief Set the month.
    //! \param[in] newMonth  New month to be set.
    //----------------------------------------
    void setMonth(const uint8_t newMonth) { this->m_month = newMonth; }

    //========================================
    //! \brief Set the year.
    //! \param[in] newYear  New year to be set.
    //----------------------------------------
    void setYear(const uint8_t newYear) { this->m_year = newYear; }

    //========================================
    //! \brief Set the cnt1 value.
    //! \param[in] newCnt1  New cnt1 value to be set.
    //----------------------------------------
    void setCnt1(const uint8_t newCnt1) { this->m_cnt1 = newCnt1; }

    //========================================
    //! \brief Set the cnt0 value.
    //! \param[in] newCnt0  New cnt0 value to be set.
    //----------------------------------------
    void setCnt0(const uint8_t newCnt0) { this->m_cnt0 = newCnt0; }

    //========================================
    //! \brief Set the null value.
    //! \param[in] newNull  New null value to be set.
    //----------------------------------------
    void setNull(const uint16_t newNull) { this->m_null = newNull; }

protected:
    uint8_t m_month; //!< Month entry of the serial number
    uint8_t m_year; //!< Year entry of the serial number
    uint8_t m_cnt1;
    uint8_t m_cnt0;
    uint16_t m_null;
}; // SerialNumberIn6301

//==============================================================================
//! \brief Comparison operator for two SerialNumberIn6301 objects for equality.
//! \param[in] lhs  First (left) SerialNumberIn6301 object to be compared.
//! \param[in] rhs  Second (right) SerialNumberIn6301 object to be compared.
//! \return \c True if the contents of both SerialNumberIn6301 objects are
//!         identically. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const SerialNumberIn6301 lhs, const SerialNumberIn6301 rhs)
{
    return (lhs.getMonth() == rhs.getMonth()) && (lhs.getYear() == rhs.getYear()) && (lhs.getCnt1() == rhs.getCnt1())
           && (lhs.getCnt0() == rhs.getCnt0()) && (lhs.getNull() == rhs.getNull());
}

//==============================================================================
//! \brief Comparison operator for two SerialNumberIn6301 objects for inequality.
//! \param[in] lhs  First (left) SerialNumberIn6301 object to be compared.
//! \param[in] rhs  Second (right) SerialNumberIn6301 object to be compared.
//! \return \c True if the contents of both SerialNumberIn6301 objects are
//!         different. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const SerialNumberIn6301 lhs, const SerialNumberIn6301 rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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

//==============================================================================
//! \brief Version number in 4/4/8 bit format with its date.
//! \date Jul 30, 2013
//!
//! Version number used for FPGA and host.
//------------------------------------------------------------------------------
class Version448In6301 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    Version448In6301()          = default;
    virtual ~Version448In6301() = default;

public:
    //========================================
    //! \brief Gets the version.
    //! \return The version.
    //----------------------------------------
    uint16_t getVersion() const { return m_version; }

    //========================================
    //! \brief Gets the year.
    //! \return The year.
    //----------------------------------------
    uint16_t getYear() const { return m_year; }

    //========================================
    //! \brief Gets the month.
    //! \return The month.
    //----------------------------------------
    uint8_t getMonth() const { return m_month; }

    //========================================
    //! \brief Gets the day.
    //! \return The day.
    //----------------------------------------
    uint8_t getDay() const { return m_day; }

    //========================================
    //! \brief Gets the hour.
    //! \return The hour.
    //----------------------------------------
    uint8_t getHour() const { return m_hour; }

    //========================================
    //! \brief Gets the minute.
    //! \return The minute.
    //----------------------------------------
    uint8_t getMinute() const { return m_minute; }

public:
    //========================================
    //! \brief Set the version to the given value.
    //! \param[in] newVersion  New value for version.
    //----------------------------------------
    void setVersion(const uint16_t newVersion) { this->m_version = newVersion; }

    //========================================
    //! \brief Set the year to the given value.
    //! \param[in] newYear  New value for year.
    //----------------------------------------
    void setYear(const uint16_t newYear) { this->m_year = newYear; }

    //========================================
    //! \brief Set the month to the given value.
    //! \param[in] newMonth  New value for month.
    //----------------------------------------
    void setMonth(const uint8_t newMonth) { this->m_month = newMonth; }

    //========================================
    //! \brief Set the day to the given value.
    //! \param[in] newDay  New value for day.
    //----------------------------------------
    void setDay(const uint8_t newDay) { this->m_day = newDay; }

    //========================================
    //! \brief Set the hour to the given value.
    //! \param[in] newHour  New value for hour.
    //----------------------------------------
    void setHour(const uint8_t newHour) { this->m_hour = newHour; }

    //========================================
    //! \brief Set the minute to the given value.
    //! \param[in] newMinute  New value for minute.
    //----------------------------------------
    void setMinute(const uint8_t newMinute) { this->m_minute = newMinute; }

protected:
    uint16_t m_version; //!< Version in 4bit.4bit.8bit format.
    uint16_t m_year; //!< year of this version
    uint8_t m_month; //!< month of this version
    uint8_t m_day; //!< day of this version
    uint8_t m_hour; //!< hour of this version
    uint8_t m_minute; //!< minute of this version
}; // Version448In6301

//==============================================================================
//! \brief Comparison operator for two Version448In6301In6301 objects for equality.
//! \param[in] lhs  First (left) Version448In6301 object to be compared.
//! \param[in] rhs  Second (right) Version448In6301 object to be compared.
//! \return \c True if the contents of both Version448In6301 objects are
//!         identically. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const Version448In6301& lhs, const Version448In6301& rhs)
{
    return (lhs.getVersion() == rhs.getVersion()) && (lhs.getYear() == rhs.getYear())
           && (lhs.getMonth() == rhs.getMonth()) && (lhs.getDay() == rhs.getDay()) && (lhs.getHour() == rhs.getHour())
           && (lhs.getMinute() == rhs.getMinute());
}

//==============================================================================
//! \brief Comparison operator for two Version448In6301 objects for inequality.
//! \param[in] lhs  First (left) Version448In6301 object to be compared.
//! \param[in] rhs  Second (right) Version448In6301 object to be compared.
//! \return \c True if the contents of both Version448In6301 objects are
//!         different. \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Version448In6301& lhs, const Version448In6301& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

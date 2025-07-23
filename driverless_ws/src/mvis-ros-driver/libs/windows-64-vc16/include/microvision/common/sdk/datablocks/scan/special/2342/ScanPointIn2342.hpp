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
//! \date Jan 18, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scan point in the format contained in a processed low bandwidth MOVIA scan.
//------------------------------------------------------------------------------
class ScanPointIn2342 final
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScanPointIn2342() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScanPointIn2342(const ScanPointIn2342& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned
    //! \return this
    //----------------------------------------
    ScanPointIn2342& operator=(const ScanPointIn2342& src) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ScanPointIn2342() = default;

public: // getter
    //========================================
    //! \brief Get the radial distance of this scan point in [cm].
    //!
    //! \return The radial distance of the scan point.
    //----------------------------------------
    uint16_t getRadialDistanceInCentimeter() const { return m_radialDistanceInCentimeter; }

    //========================================
    //! \brief Get the existence measure of this scan point.
    //!
    //! \return The existence measure of the scan point.
    //----------------------------------------
    uint16_t getExistenceMeasure() const { return m_existenceMeasure; }

    //========================================
    //! \brief Get the intensity of this scan point in photon counts.
    //!
    //! \return The intensity of the scan point.
    //----------------------------------------
    uint16_t getIntensity() const { return m_intensity; }

    //========================================
    //! \brief Get the pulse width of this scan point.
    //!
    //! \return The pulse width of the scan point.
    //!
    //! \note The pulse with is in fixed comma notation Q4.12.
    //! The first 4 bits describe the integer part.
    //! the last 12 bits describe the fractional part.
    //----------------------------------------
    uint16_t getPulseWidth() const { return m_pulseWidth; }

public:
    //========================================
    //! \brief Set the radial distance of this scan point in [cm].
    //!
    //! \param[in] radialDistance  The new start radial distance.
    //----------------------------------------
    void setRadialDistanceInCentimeter(uint16_t radialDistance) { m_radialDistanceInCentimeter = radialDistance; }

    //========================================
    //! \brief Set the existence measure of this scan point.
    //!
    //! \param[in] existenceMeasure  The new start existence measure.
    //----------------------------------------
    void setExistenceMeasure(uint16_t existenceMeasure) { m_existenceMeasure = existenceMeasure; }

    //========================================
    //! \brief Set the intensity of this scan point in photon counts.
    //!
    //! \param[in] intensity  The new start intensity.
    //----------------------------------------
    void setIntensity(uint16_t intensity) { m_intensity = intensity; }

    //========================================
    //! \brief Set the pulse width of this scan point. [unit less]
    //!
    //! \param[in] pulseWidth  The new pulse width.
    //!
    //! \note The pulse with is in fixed comma notation Q4.12.
    //! The first 4 bits describe the integer part.
    //! the last 12 bits describe the fractional part.
    //----------------------------------------
    void setPulseWidth(uint16_t pulseWidth) { m_pulseWidth = pulseWidth; }

private:
    //========================================
    //! The radial distance of echo in ISMC in [cm].
    //!
    //! The invalid value is 0xFFFF.
    //----------------------------------------
    uint16_t m_radialDistanceInCentimeter{};

    //========================================
    //! The existence measure.
    //!
    //! The invalid value is 0xFFFF.
    //! Existence measure information = 1/(65535-1) * m_existenceMeasure.
    //----------------------------------------
    uint16_t m_existenceMeasure{};

    //========================================
    //! The raw intensity in photon counts.
    //----------------------------------------
    uint16_t m_intensity{};

    //========================================
    //! The pulse width ratio in fixed comma notation Q4.12
    //!
    //! First 4 bits describe the integer part.
    //! Last 12 bits describe the fractional part.
    //! 0x0000 if the measured receive pulse is smaller than estimable.
    //----------------------------------------
    uint16_t m_pulseWidth{};

}; // ScanPointIn2342

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScanPointIn2342& lhs, const ScanPointIn2342& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScanPointIn2342& lhs, const ScanPointIn2342& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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

#include <microvision/common/sdk/datablocks/scan/special/2342/ScanPointIn2342.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Scan point row data used by a processed low bandwidth MOVIA scan.
//------------------------------------------------------------------------------
class ScanPointRowIn2342 final
{
public:
    static constexpr uint8_t nbOfEchos{3}; //!< The size of the point including the echos.
    static constexpr uint8_t nbOfPoints{128}; //!< The number of the points per row.

public:
    using EchoArray  = std::array<ScanPointIn2342, nbOfEchos>;
    using PointArray = std::array<EchoArray, nbOfPoints>;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScanPointRowIn2342() = default;

    //========================================
    //! \brief Default copy-constructor.
    //!
    //! \param[in] src  Object to create a copy from.
    //----------------------------------------
    ScanPointRowIn2342(const ScanPointRowIn2342& src) = default;

    //========================================
    //! \brief Default assignment operator.
    //!
    //! \param[in] src  Object to be assigned
    //! \return this
    //----------------------------------------
    ScanPointRowIn2342& operator=(const ScanPointRowIn2342& src) = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ScanPointRowIn2342() = default;

public: // getter
    //========================================
    //! \brief Get the start timestamp.
    //!
    //! \return The start timestamp of the scan.
    //----------------------------------------
    const NtpTime& getTimestampStart() const { return m_timestampStart; }

    //========================================
    //! \brief Get the timestamp offset in [ns].
    //!
    //! \return The timestamp offset of the scan.
    //----------------------------------------
    uint32_t getTimestampOffsetInNanoseconds() const { return m_timeOffsetInNanoseconds; }

    //========================================
    //! \brief Get the scan points of this row of the scan.
    //!
    //! \return The scan points of this row of the scan.
    //----------------------------------------
    const PointArray& getScanPoints() const { return m_scanPoints; }

    //========================================
    //! \brief Get the scan points of this row of the scan.
    //!
    //! \return The scan points of this row of the scan.
    //----------------------------------------
    PointArray& getScanPoints() { return m_scanPoints; }

public: // setter
    //========================================
    //! \brief Set the start timestamp.
    //!
    //! \param[in] timestamp  The new start timestamp.
    //----------------------------------------
    void setTimestampStart(const NtpTime& timestamp) { m_timestampStart = timestamp; }

    //========================================
    //! \brief Set the timestamp offset in [ns].
    //!
    //! \param[in] timeOffset  The new timestamp offset.
    //----------------------------------------
    void setTimestampOffsetInNanoseconds(uint32_t timeOffset) { m_timeOffsetInNanoseconds = timeOffset; }

    //========================================
    //! \brief Set the scan points of this row of the scan.
    //!
    //! \param[in] scanPoints  The new scanPoints.
    //----------------------------------------
    void setScanPoints(const PointArray& scanPoints) { m_scanPoints = scanPoints; }

private:
    NtpTime m_timestampStart{}; //!< The absolute timestamp of the scan start of this row.
    uint32_t m_timeOffsetInNanoseconds{}; //!< The time offset in ns of this row.
    PointArray m_scanPoints{}; //!< The array of scan points in this row.

}; // ScanPointRowIn2342

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ScanPointRowIn2342& lhs, const ScanPointRowIn2342& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const ScanPointRowIn2342& lhs, const ScanPointRowIn2342& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

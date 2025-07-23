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
//! \date Aug 2, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2208.hpp>

#include <microvision/common/sdk/NtpTime.hpp>

#include <vector>
#include <istream>
#include <ostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class SubScanIn2208 final
{
public:
    static const uint8_t nbOfReserved   = 6;
    static const uint16_t maxNbOfPoints = 9 * 581;

public:
    using PointVector = std::vector<ScanPointIn2208>;
    enum class Flags : uint8_t
    {
        Laser = 0x01U //!< Id of the used laser.  0 = laser 0,  1 = laser 1 (if available)
        // reserved
    }; // Flags

public:
    SubScanIn2208();
    virtual ~SubScanIn2208();

public:
    //! Equality predicate
    bool operator==(const SubScanIn2208& other) const;

    bool operator!=(const SubScanIn2208& other) const;

public:
    virtual std::streamsize getSerializedSize() const
    {
        return 36 + ScanPointIn2208::getSerializedSize_static() * std::streamsize(scanPoints.size());
    }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    NtpTime getStartScanTimestamp() const { return startScanTimestamp; }
    NtpTime getEndScanTimestamp() const { return endScanTimestamp; }
    int16_t getStartScanAngle() const { return startScanAngle; }
    int16_t getEndScanAngle() const { return endScanAngle; }
    uint8_t getFlags() const { return flags; }
    uint8_t getMirrorSide() const { return mirrorSide; }
    float getMirrorTiltDeprecated() const { return mirrorTiltDeprecated; }
    uint16_t getMirrorTilt() const { return mirrorTilt; }
    uint8_t getReserved(const uint8_t idx) const { return reserved[idx]; }
    uint32_t getNbOfPoints() const { return uint16_t(scanPoints.size()); }

    const PointVector& getScanPoints() const { return scanPoints; }
    PointVector& getScanPoints() { return scanPoints; }

public:
    void setStartScanTimestamp(const NtpTime newStartScanTimestamp) { startScanTimestamp = newStartScanTimestamp; }
    void setEndScanTimestamp(const NtpTime newEndScanTimestamp) { endScanTimestamp = newEndScanTimestamp; }
    void setStartScanAngle(const int16_t newStartScanAngle) { startScanAngle = newStartScanAngle; }
    void setEndScanAngle(const int16_t newEndScanAngle) { endScanAngle = newEndScanAngle; }
    void setFlags(const uint8_t newFlags) { flags = newFlags; }
    void setMirrorSide(const uint8_t newMirrorSide) { mirrorSide = newMirrorSide; }
    void setMirrorTiltDeprecated(const float newMirrorTiltDeprecated)
    {
        mirrorTiltDeprecated = newMirrorTiltDeprecated;
    }
    void setMirrorTilt(const uint16_t newMirrorTilt) { mirrorTilt = newMirrorTilt; }
    void setReserved(const uint8_t idx, const uint8_t newReserved) { reserved[idx] = newReserved; }

protected:
    NtpTime startScanTimestamp{}; //!< Ntp time-stamp of scan start. The timer starts with 0 at power-up.
    NtpTime endScanTimestamp{}; //!< Ntp time-stamp of scan end. The timer starts with 0 at power-up.
    int16_t startScanAngle{0}; //!< Scan start angle given in angle ticks [ticks].
    int16_t endScanAngle{0}; //!< Scan end angle given in angle ticks [ticks].
    uint8_t flags{}; //!< Flags in the SubScan.
    uint8_t mirrorSide{
        0}; //!< To code different mirror planes, having different tilt angles, could be more than just two.
    float mirrorTiltDeprecated{0.0F}; //!< Deprecated mirror tilt. [deg]
    uint16_t mirrorTilt{0}; //!< Mirror tilt, see above, here Unit in 1/500 deg. [1/500 deg]
    uint8_t reserved[nbOfReserved]{}; //!< Reserved.
    // uint16_t nbOfPoints
    PointVector scanPoints{}; //!< Vector of ScanPoints in this SubScan.
}; // SubScanIn2208

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

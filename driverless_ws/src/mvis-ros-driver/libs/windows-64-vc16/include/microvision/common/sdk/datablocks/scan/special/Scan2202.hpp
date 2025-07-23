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
//! \date Jan 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2202.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of scan points in the format of the LUX3 firmware
//!
//! Scan data available from MVIS LUX laser scanners (not available for MVIS LUX prototypes).
//! Each scan data block starts with a header followed by the scan point list.
//! The data is encoded in little-endian format!
//!
//! For angle information the unit angle ticks is used. A MVIS LUX typically uses 11520 ticks per rotation
//! (see also Angle ticks per rotation below).
//! Thus the angular resolution is 1/32°. This value is needed to convert angle ticks.
//!
//! Angles are given in the ISO 8855 / DIN 70000 scanner coordinate system.
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2202 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ScanPointVector = std::vector<ScanPointIn2202>;

public:
    enum class ScannerStatus : uint16_t
    {
        MotorOn            = 0x0001U, //!< Motor on.
        LaserOn            = 0x0002U, //!< Laser on.
        InternalFeedback   = 0x0004U, //!< Internal feedback.
        SetFreqReached     = 0x0008U, //!< Set frequency reached.
        ExtSyncSigDetected = 0x0010U, //!< External sync signal detected.
        SyncOk             = 0x0020U, //!< Sync is ok.
        SyncMaster         = 0x0040U, //!< Is sync master and not slave.
        // 0x0080 reserved
        EpwCompOn        = 0x0100U, //!< EPW compensation on.
        SystemCompOn     = 0x0200U, //!< System compensation on.
        StartPulseCompOn = 0x0400U, //!< Start pulse compensation on.
        // 0x0800 reserved
        // 0x1000 reserved
        // 0x2000 reserved
        // 0x4000 reserved
        UpsideDown = 0x8000U //!< Sensor is upside down, FPGA version >= 0x9604.
    }; // ScannerStatus

    //========================================

    enum class Flags : uint16_t
    {
        GroundLabeled = 0x0001U, //!< Ground is labeled.
        DirtLabeled   = 0x0002U, //!< Dirt is labeled.
        RainLabeled   = 0x0004U, //!< Rain is labeled.
        // bit  3, 0x0008, reserved
        // bit  4, 0x0010, internal
        // bit  5, 0x0020, internal
        // bit  6, 0x0040, internal
        // bit  7, 0x0080, reserved
        // bit  8, 0x0100, reserved
        // bit  9, 0x0200, reserved
        RearMirrorSide = 0x0400U //!< Mirror side \b 0 = front, \b 1 = rear
        // bit 11, 0x0800, reserved
        // bit 12, 0x1000, reserved
        // bit 13, 0x2000, reserved
        // bit 14, 0x4000, reserved
        // bit 15, 0x8000, reserved
    }; // Flags

    //========================================

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2202"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Scan2202();
    ~Scan2202() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    uint16_t getScanNumber() const { return m_scanNumber; }
    uint16_t getScannerStatus() const { return m_scannerStatus; }
    uint16_t getSyncPhaseOffset() const { return m_syncPhaseOffset; }

    NtpTime getStartTimestamp() const { return m_startNtp; }
    NtpTime getEndTimestamp() const { return m_endNtp; }

    uint16_t getAngleTicksPerRotation() const { return m_angleTicksPerRotation; }

    int16_t getStartAngleTicks() const { return m_startAngleTicks; }
    int16_t getEndAngleTicks() const { return m_endAngleTicks; }

    int16_t getMountingPositionYawAngleTicks() const { return m_mountingPosYawAngleTicks; }
    int16_t getMountingPositionPitchAngleTicks() const { return m_mountingPosPitchAngleTicks; }
    int16_t getMountingPositionRollAngleTicks() const { return m_mountingPosRollAngleTicks; }

    int16_t getMountingPositionCmX() const { return m_mountingPosCmX; }
    int16_t getMountingPositionCmY() const { return m_mountingPosCmY; }
    int16_t getMountingPositionCmZ() const { return m_mountingPosCmZ; }

    uint16_t getFlags() const { return m_flags; }
    bool isFlagSet(const Flags flags) const
    {
        return (m_flags & static_cast<uint16_t>(flags)) == static_cast<uint16_t>(flags);
    }

    //! Returns true if the ground detection ran on this scan.
    bool isGroundLabeled() const { return isFlagSet(Flags::GroundLabeled); }

    //! Returns true if the dirt detection ran on this scan.
    bool isDirtLabeled() const { return isFlagSet(Flags::DirtLabeled); }

    //! Returns true if the rain detection ran on this scan.
    bool isRainLabeled() const { return isFlagSet(Flags::RainLabeled); }

    //! Returns true if the scan was from a rear mirror
    bool isRearMirrorSide() const { return isFlagSet(Flags::RearMirrorSide); }

    uint16_t getNumberOfScanPoints() const { return uint16_t(m_points.size()); }

    const ScanPointVector& getScanPoints() const { return m_points; }
    ScanPointVector& getScanPoints() { return m_points; }

public: // setter
    void setScanNumber(const uint16_t newScanNumber) { this->m_scanNumber = newScanNumber; }
    void setScannerStatus(const uint16_t newScannerStatus) { this->m_scannerStatus = newScannerStatus; }
    void setSyncPhaseOffset(const uint16_t newSyncPhaseOffset) { this->m_syncPhaseOffset = newSyncPhaseOffset; }

    void setStartTimestamp(const NtpTime newStartTimestamp) { this->m_startNtp = newStartTimestamp; }
    void setEndTimestamp(const NtpTime newEndTimestamp) { this->m_endNtp = newEndTimestamp; }

    void setAngleTicksPerRotation(const uint16_t newAngleTicksPerRotation)
    {
        this->m_angleTicksPerRotation = newAngleTicksPerRotation;
    }

    void setStartAngleTicks(const int16_t newStartAngleTicks) { this->m_startAngleTicks = newStartAngleTicks; }
    void setEndAngleTicks(const int16_t newEndAngleTicks) { this->m_endAngleTicks = newEndAngleTicks; }

    void setMountingPositionYawAngleTicks(const int16_t newMountingPositionYawAngleTicks)
    {
        this->m_mountingPosYawAngleTicks = newMountingPositionYawAngleTicks;
    }
    void setMountingPositionPitchAngleTicks(const int16_t newMountingPositionPitchAngleTicks)
    {
        this->m_mountingPosPitchAngleTicks = newMountingPositionPitchAngleTicks;
    }
    void setMountingPositionRollAngleTicks(const int16_t newMountingPositionRollAngleTicks)
    {
        this->m_mountingPosRollAngleTicks = newMountingPositionRollAngleTicks;
    }

    void setMountingPositionCmX(const int16_t newMountingPositionX) { this->m_mountingPosCmX = newMountingPositionX; }
    void setMountingPositionCmY(const int16_t newMountingPositionY) { this->m_mountingPosCmY = newMountingPositionY; }
    void setMountingPositionCmZ(const int16_t newMountingPositionZ) { this->m_mountingPosCmZ = newMountingPositionZ; }

    void setFlags(const uint16_t newFlags) { m_flags = newFlags; }
    void setFlag(const Flags flag) { m_flags |= static_cast<uint16_t>(flag); }
    void clearFlag(const Flags flag) { m_flags = static_cast<uint16_t>(m_flags & (~static_cast<uint32_t>(flag))); }
    void setFlag(const Flags flag, const bool value) { value ? setFlag(flag) : clearFlag(flag); }

    //! Set the Scan has ground labeled.
    void setGroundLabeled(const bool isGroundLabeled = true) { setFlag(Flags::GroundLabeled, isGroundLabeled); }

    //! Set the Scan has dirt labeled.
    void setDirtLabeled(const bool isDirtLabeled = true) { setFlag(Flags::DirtLabeled, isDirtLabeled); }

    //! Set the Scan has rain labeled.
    void setRainLabeled(const bool isRainLabeled = true) { setFlag(Flags::RainLabeled, isRainLabeled); }

    void setNumberOfScanPoints(const uint16_t newNumberOfScanPoints) { this->m_points.resize(newNumberOfScanPoints); }
    void setScanPoint(const uint32_t idx, const ScanPointIn2202& sp) { this->m_points[idx] = sp; }

protected:
    //! check if start if lower then end time for this scan, used when importing
    bool timeCheck() const { return m_startNtp < m_endNtp; }

    bool nbOfAngleTicksPerRotationCheck() const;

protected:
    static const uint16_t expectedNbOfAngleTicksPerRotation{11520};

protected:
    static constexpr const char* loggerId = "microvision::common::sdk::Scan2202";
    static microvision::common::logging::LoggerSPtr logger;

protected:
    uint16_t m_scanNumber{0}; //!< The number of this scan.
    uint16_t m_scannerStatus{0}; //!< The status as recieved from FPGA.

    //! The Phase difference (conversion factor 409.6 ns) bwn. sync sig and mirror cross sync angle
    uint16_t m_syncPhaseOffset{0};
    NtpTime m_startNtp{0}; //!< NTP time when the first measurement was done.
    NtpTime m_endNtp{0}; //!< NTP time when the last measurement was done.
    uint16_t m_angleTicksPerRotation{
        0}; //!< The number of angle ticks per rotation. (for conv. angle ticks to rad or deg)
    int16_t m_startAngleTicks{0}; //!< Start angle in angle ticks of this scan. [ticks]
    int16_t m_endAngleTicks{0}; //!< End angle in angle ticks of this scan. [ticks]
    // m_points.size() uint16_t
    int16_t m_mountingPosYawAngleTicks{0}; //!<[ticks]  - in vehicle coord system
    int16_t m_mountingPosPitchAngleTicks{0}; //!<[ticks]  - in vehicle coord system
    int16_t m_mountingPosRollAngleTicks{0}; //!<[ticks]  - in vehicle coord system
    int16_t m_mountingPosCmX{0}; //!< [cm]  - in vehicle coord system
    int16_t m_mountingPosCmY{0}; //!<[cm]  - in vehicle coord system
    int16_t m_mountingPosCmZ{0}; //!< [cm]  - in vehicle coord system
    uint16_t m_flags{0}; //!<
    ScanPointVector m_points{}; //!<
}; // Scan2202Container

//==============================================================================

bool operator==(const Scan2202& lhs, const Scan2202& rhs);
bool operator!=(const Scan2202& lhs, const Scan2202& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

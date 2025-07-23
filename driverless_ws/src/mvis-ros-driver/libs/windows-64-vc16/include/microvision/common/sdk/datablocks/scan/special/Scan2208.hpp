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
//! \date Jan 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/scan/special/SubScanIn2208.hpp>
#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/datablocks/scan/ScannerInfo.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Raw scan point list for Scala and Minilux
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2208 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfThresholds = 7;
    static const uint8_t nbOfReserved   = 12;
    static const uint8_t maxSubScans    = 2;

public: // type declarations
    using ReservedArray  = std::array<uint8_t, nbOfReserved>;
    using ThresholdArray = std::array<uint16_t, nbOfThresholds>;

public:
    enum class ScannerStatusBits : uint16_t
    {
        MotorOn         = 0x0001U,
        Laser1On        = 0x0002U,
        DemoModeOn      = 0x0004U,
        FrequencyLocked = 0x0008U,
        // reserved 0x0010
        // reserved 0x0020
        // reserved 0x0040
        Laser2On                      = 0x0080U,
        MotorDirectionConterClockwise = 0x0100U
        // rest is reserved
    }; // ScannerStatusBits

    //========================================

    enum class Flags : uint16_t
    {
        GroundLabeled = 0x0001U, //!< Ground detection was performed.
        DirtLabeled   = 0x0002U, //!< Dirt detection was performed.
        RainLabeled   = 0x0004U, //!< Clutter detection was performed.
        // bit  3, 0x0008, reserved
        // bit  4, 0x0010, internal
        // bit  5, 0x0020, internal
        // bit  6, 0x0040, internal
        // bit  7, 0x0080, reserved
        // bit  8, 0x0100, reserved
        Fusion = 0x0200U, //!< Scan is a fusion of scans.
        // bit 10, 0x0400, reserved
        VehicleCoordinates
        = 0x0800U //!< Scan point coordinate system: 0 = scanner coordinates, 1 = vehicle / reference coordinates.
        // bit 12, 0x1000, reserved
        // bit 13, 0x2000, reserved
        // bit 14, 0x4000, reserved
        // bit 15, 0x8000, reserved
    }; // Flags

public:
    using SubScanVector          = std::vector<SubScanIn2208>;
    using MountingPositionIn2208 = MountingPosition<int16_t>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2208"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Scan2208();
    ~Scan2208() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint16_t getScanNumber() const { return m_scanNumber; }
    microvision::common::sdk::ScannerType getScannerType() const { return m_scannerType; }
    uint16_t getScannerStatus() const { return m_scannerStatus; }
    uint16_t getAngleTicksPerRotation() const { return m_angleTicksPerRotation; }

    uint32_t getProcessingFlags() const { return m_processingFlags; }
    bool isProcessingFlagSet(const Flags flags) const
    {
        return (m_processingFlags & static_cast<uint32_t>(flags)) == static_cast<uint32_t>(flags);
    }

    //! Returns true if the ground detection ran on this scan.
    bool isGroundLabeled() const { return isProcessingFlagSet(Flags::GroundLabeled); }

    //! Returns true if the dirt detection ran on this scan.
    bool isDirtLabeled() const { return isProcessingFlagSet(Flags::DirtLabeled); }

    //! Returns true if the rain detection ran on this scan.
    bool isRainLabeled() const { return isProcessingFlagSet(Flags::RainLabeled); }

    //! Returns true if the scan was calculated from the fusion of several scans
    bool isFusionScan() const { return isProcessingFlagSet(Flags::Fusion); }

    //! Returns true if the scan points are given in vehicle coordinates, or false if the scan points are given in the Laserscanner coordinate system.
    bool isVehicleCoordinates() const { return isProcessingFlagSet(Flags::VehicleCoordinates); }

    const MountingPositionIn2208& getMountingPosition() const { return m_mountingPosition; }
    MountingPositionIn2208& getMountingPosition() { return m_mountingPosition; }
    uint16_t getThreshold(const uint8_t idx) const { return m_thresholds.at(idx); }
    uint8_t getReserved(const uint8_t idx) const { return m_reserved.at(idx); }
    uint8_t getDeviceId() const { return m_deviceId; }
    uint8_t getNbOfSubScans() const { return uint8_t(m_subScans.size()); }

    const SubScanVector& getSubScans() const { return this->m_subScans; }
    SubScanVector& getSubScans() { return this->m_subScans; }

public:
    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }
    void setScannerType(const microvision::common::sdk::ScannerType newScannerType) { m_scannerType = newScannerType; }
    void setScannerStatus(const uint16_t newScannerStatus) { m_scannerStatus = newScannerStatus; }

    void setProcessingFlags(const uint32_t newFlags) { m_processingFlags = newFlags; }
    void setProcessingFlag(const Flags flag) { m_processingFlags |= static_cast<uint32_t>(flag); }
    void clearProcessingFlag(const Flags flag) { m_processingFlags &= ~static_cast<uint32_t>(flag); }
    void setProcessingFlag(const Flags flag, const bool value)
    {
        value ? setProcessingFlag(flag) : clearProcessingFlag(flag);
    }

    //! Set the Scan has ground labeled.
    void setGroundLabeled(const bool isGroundLabeled = true)
    {
        setProcessingFlag(Flags::GroundLabeled, isGroundLabeled);
    }

    //! Set the Scan has dirt labeled.
    void setDirtLabeled(const bool isDirtLabeled = true) { setProcessingFlag(Flags::DirtLabeled, isDirtLabeled); }

    //! Set the Scan has rain labeled.
    void setRainLabeled(const bool isRainLabeled = true) { setProcessingFlag(Flags::RainLabeled, isRainLabeled); }

    //! Set whether the scan was calculated from the fusion of several scans
    void setFusionScan(const bool isFusionScan = true) { setProcessingFlag(Flags::Fusion, isFusionScan); }

    //! Set whether the scanpoints are given in vehicle coordinates
    void setVehicleCoordinates(const bool inVehicleCoordinates = true)
    {
        setProcessingFlag(Flags::VehicleCoordinates, inVehicleCoordinates);
    }

    void setAngleTicksPerRotation(const uint16_t newAngleTicksPerRotation)
    {
        m_angleTicksPerRotation = newAngleTicksPerRotation;
    }
    void setThreshold(const uint8_t idx, const uint16_t newThresholds) { m_thresholds.at(idx) = newThresholds; }
    void setMountingPosition(const MountingPositionIn2208 newMountingPosition)
    {
        m_mountingPosition = newMountingPosition;
    }
    void setReserved(const uint8_t idx, const uint8_t newReserved) { m_reserved.at(idx) = newReserved; }
    //void setSubScans(SubScanVector newSubScanVector) {m_subScans = newSubScanVector; }
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }

protected:
    uint16_t m_scanNumber{0}; //!< The number of this scan.
    ScannerType m_scannerType{ScannerType::Invalid}; //!< The scanner type which is used.
    uint16_t m_scannerStatus{0}; //!< The scanner status flags.
    uint16_t m_angleTicksPerRotation{0}; //!< Number of counter ticks per mirror rotation. [tick]
    uint32_t m_processingFlags{0}; //!< The processing flags.
    MountingPositionIn2208
        m_mountingPosition{}; //!< Mounting position of the scanner relative to the reference coordinate system
    ThresholdArray m_thresholds{{}}; //!< Array of threshold voltages [mV], invalid is 0xFFFF
    ReservedArray m_reserved{{}}; //!< Reserved.
    uint8_t m_deviceId{0}; //!< Id of the device.
    // uint8_t number of sub scans.
    SubScanVector m_subScans{}; //!< Vector of SubScans in this Scan.
}; // Scan2208Container

//==============================================================================

//==============================================================================

bool operator==(const Scan2208& lhs, const Scan2208& rhs);
bool operator!=(const Scan2208& lhs, const Scan2208& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
//! \date an 14, 2018
//------------------------------------------------------------------------------
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/scan/special/ScannerInfoIn2205.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2205.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU scan data:
//! List of scan points in the new proposed format of class Scan (Generic)
//!
//! Scan data available from FUSION SYSTEM and AppBase2 (ECU) is sent as data type 0x2205
//! using MVIS FUSION SYSTEM version 2.2 and later and MVIS LaserView 1.6 and later.
//! Please see data type 0x2204 for earlier versions.
//! Each scan data block starts with a header followed by the scanner info list and the scan point list.
//! Each scan point has a device ID that refers to a sensor in the sensor info list.
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2205 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ScannerInfoVector = std::vector<ScannerInfoIn2205>;
    using ScanPointVector   = std::vector<ScanPointIn2205>;

public:
    //! Mask for all processing status flags ("...labeled").
    static constexpr uint32_t labelFlagMask{0xFFFFFFC0U};

    enum class Flags : uint32_t
    {
        GroundLabeled       = 0x00000001U, //!< Bit  0: ground labeled
        DirtLabeled         = 0x00000002U, //!< Bit  1: dirt labeled
        RainLabeled         = 0x00000004U, //!< Bit  2: rain labeled
        CoverageLabeled     = 0x00000008U, //!< Bit  3: coverage labeled
        BackgroundLabeled   = 0x00000010U, //!< Bit  4: background labeled
        ReflectorLabeled    = 0x00000020U, //!< Bit  5: reflector labeled
        ApduReductionActive = 0x00000080U, //!< Bit  7: APD reduction active
        UpsideDown          = 0x00000100U, //!< Bit  8: upside down
        Fusion              = 0x00000200U, //!< Bit  9: scan fusion
        RearMirrorSide      = 0x00000400U, //!< Bit 10: mirror side, 0 = front, 1 = rear
        //! Bit 11: scan point coordinate system: 0 = scanner coordinates, 1 = vehicle / reference coordinates
        VehicleCoordinates = 0x00000800U,
        ObjIdAsSegId       = 0x00001000U, //!< Bit 12: segment ID of scan points is used as object ID
        RayAngleCorrected  = 0x00002000U, //!< Bit 13: ray angle corrected (window refraction, etc.)
        // 0x00004000 reserved for rotation order
        // 0x00008000 reserved for rotation order
        // 0x00010000 reserved for rotation order
        HorizontalIdAsSegId = 0x00020000U //!< Bit 17: scan points segmentId contains horizontal id for MOVIA scan
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2205"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Scan2205();
    ~Scan2205() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    NtpTime getStartTimestamp() const { return m_scanStartTime; }
    uint32_t getEndTimeOffset() const { return m_endTimeOffset; }

    uint32_t getFlags() const { return m_flags; }
    bool isFlagSet(const Flags flags) const
    {
        return (m_flags & static_cast<uint32_t>(flags)) == static_cast<uint32_t>(flags);
    }

    //! Returns true if the ground detection ran on this scan.
    bool isGroundLabeled() const { return isFlagSet(Flags::GroundLabeled); }

    //! Returns true if the dirt detection ran on this scan.
    bool isDirtLabeled() const { return isFlagSet(Flags::DirtLabeled); }

    //! Returns true if the rain detection ran on this scan.
    bool isRainLabeled() const { return isFlagSet(Flags::RainLabeled); }

    //! Returns true if the cover detection ran on this scan.
    bool isCoverageLabeled() const { return isFlagSet(Flags::CoverageLabeled); }

    //! Returns true if the background detection ran on this scan.
    bool isBackgroundLabeled() const { return isFlagSet(Flags::BackgroundLabeled); }

    //! Returns true if the reflector detection ran on this scan.
    bool isReflectorLabeled() const { return isFlagSet(Flags::ReflectorLabeled); }

    //! Returns true if the APDU reduction is active.
    bool isApduReductionActive() const { return isFlagSet(Flags::ApduReductionActive); }

    //! Returns true if the scan originated from an upside down mounted scanner.
    bool isUpsideDown() const { return isFlagSet(Flags::UpsideDown); }

    //! Returns true if the scan was calculated from the fusion of several scans
    bool isFusionScan() const { return isFlagSet(Flags::Fusion); }

    //! Returns true if the scan was from a rear mirror
    bool isRearMirrorSide() const { return isFlagSet(Flags::RearMirrorSide); }

    //! Returns true if the scan points are given in vehicle coordinates, or false if the scan points are given
    //! in the laser scanner coordinate system.
    bool isVehicleCoordinates() const { return isFlagSet(Flags::VehicleCoordinates); }

    //! Returns true if the scan points' segment ID is used as object ID.
    bool isSegIDUsedAsObjID() const { return isFlagSet(Flags::ObjIdAsSegId); }

    //! Returns true the ray angle has been corrected (Window refraction compensation, etc.).
    bool isRayAngleCorrected() const { return isFlagSet(Flags::RayAngleCorrected); }

    //========================================
    //! \brief Returns true if the scan points have been imported from a Scan2340 (MOVIA scan) and the segment id contains the horizontal id.
    //!
    //! \return \c True if scan is imported from Scan2340, \c false otherwise.
    //----------------------------------------
    bool isSegIdUsedAsHorizontalId() const { return isFlagSet(Flags::HorizontalIdAsSegId); }

    //========================================
    //! \brief Set whether the segment id of the scan points contains the horizontal id from a MOVIA scan.
    //!
    //! \param[in] segmentIdIshorizontalId  \c True if scan is imported from Scan2340, \c false otherwise.
    //----------------------------------------
    void setSegIdUsedAsHorizontalId(const bool segmentIdIshorizontalId = true)
    {
        setFlag(Flags::HorizontalIdAsSegId, segmentIdIshorizontalId);
    }

    uint16_t getScanNumber() const { return m_scanNumber; }

    uint8_t getNumberOfScannerInfos() const { return uint8_t(m_scannerInfos.size()); }

    const ScannerInfoVector& getScannerInfos() const { return m_scannerInfos; }
    ScannerInfoVector& getScannerInfos() { return m_scannerInfos; }

    uint16_t getNumberOfScanPoints() const { return uint16_t(m_scanPoints.size()); }

    uint8_t getReserved0() const { return m_reserved0; }
    uint16_t getReserved1() const { return m_reserved1; }

    const ScanPointVector& getScanPoints() const { return m_scanPoints; }
    ScanPointVector& getScanPoints() { return m_scanPoints; }

public:
    void setStartTimestamp(const NtpTime newScanStartTime) { m_scanStartTime = newScanStartTime; }
    void setEndTimeOffset(const uint32_t newEndTimeOffset) { m_endTimeOffset = newEndTimeOffset; }

    void setFlags(const uint32_t newFlags) { m_flags = newFlags; }
    void setFlag(const Flags flag) { m_flags |= static_cast<uint32_t>(flag); }
    void clearFlag(const Flags flag) { m_flags &= ~static_cast<uint32_t>(flag); }
    void setFlag(const Flags flag, const bool value) { value ? setFlag(flag) : clearFlag(flag); }

    //========================================
    //! \brief Removes all processing status flags ("...labeled") from the Scan object, but leaves the other flags
    //! unchanged.
    //!
    //! Note: This only changes the flags of the Scan object, not the flags in the individual scan points! The flags
    //! in the individual points are unchanged and must be modified in a manual iteration loop.
    //----------------------------------------
    void clearLabelFlags() { m_flags &= labelFlagMask; }

    //! Set the Scan has ground labeled.
    void setGroundLabeled(const bool isGroundLabeled = true) { setFlag(Flags::GroundLabeled, isGroundLabeled); }

    //! Set the Scan has dirt labeled.
    void setDirtLabeled(const bool isDirtLabeled = true) { setFlag(Flags::DirtLabeled, isDirtLabeled); }

    //! Set the Scan has rain labeled.
    void setRainLabeled(const bool isRainLabeled = true) { setFlag(Flags::RainLabeled, isRainLabeled); }

    //! Set the Scan has coverage labeled.
    void setCoverageLabeled(const bool isCoverageLabeled = true) { setFlag(Flags::CoverageLabeled, isCoverageLabeled); }

    //! Set the Scan has background labeled.
    void setBackgroundLabeled(const bool isBackgroundLabeled = true)
    {
        setFlag(Flags::BackgroundLabeled, isBackgroundLabeled);
    }

    //! Set the Scan has reflector labeled.
    void setReflectorLabeled(const bool isReflectorLabeled = true)
    {
        setFlag(Flags::ReflectorLabeled, isReflectorLabeled);
    }

    //! Set the flag which says this scan originated from an upside down mounted scanner.
    void setUpsideDown(const bool isUpsideDown = true) { setFlag(Flags::UpsideDown, isUpsideDown); }

    //! Set the flag which says APDU reduction is active..
    void setApduReductionActive(const bool isApduReductionActive = true)
    {
        setFlag(Flags::ApduReductionActive, isApduReductionActive);
    }

    //! Set whether the scan was calculated from the fusion of several scans
    void setFusionScan(const bool isFusionScan = true) { setFlag(Flags::Fusion, isFusionScan); }

    //! Set whether the scanpoints are given in vehicle coordinates
    void setVehicleCoordinates(const bool inVehicleCoordinates = true)
    {
        setFlag(Flags::VehicleCoordinates, inVehicleCoordinates);
    }

    //! Set whether the scan was from a rear mirror
    void setRearMirrorSide(const bool isRearMirrorSide = true) { setFlag(Flags::RearMirrorSide, isRearMirrorSide); }

    //! Set whether the scan points' segment ID is used as object ID.
    void setSegIDUsedAsObjID(const bool isSegIdUsedAsObjId = true) { setFlag(Flags::ObjIdAsSegId, isSegIdUsedAsObjId); }

    /// Set whether the ray angle has been corrected (Window refraction compensation, etc.)
    void setRayAngleCorrected(const bool isRayAngleCorrected = true)
    {
        setFlag(Flags::RayAngleCorrected, isRayAngleCorrected);
    }

    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }

    void setReserved0(const uint8_t newReserved0) { m_reserved0 = newReserved0; }
    void setReserved1(const uint16_t newReserved1) { m_reserved1 = newReserved1; }

    void setScannerInfos(const ScannerInfoVector& newScannerInfos) { m_scannerInfos = newScannerInfos; }
    void setScannerInfos(ScannerInfoVector&& newScannerInfos) { m_scannerInfos = std::move(newScannerInfos); }

    void setScanPoints(const ScanPointVector& newScanPoints) { m_scanPoints = newScanPoints; }
    void setScanPoints(ScanPointVector&& newScanPoints) { m_scanPoints = std::move(newScanPoints); }

protected:
    NtpTime m_scanStartTime{}; //!< NtpTime when the first measurement was done.
    uint32_t m_endTimeOffset{}; //!< Time difference between last and first measurement in us. [us]
    uint32_t m_flags{0}; //!< Scan flags.
    uint16_t m_scanNumber{0}; //!< The number of this scan. The number will be increased from scan to scan.
    //	uint16_t number of scan points;
    //	uint8_t number of scan infos;
    uint8_t m_reserved0{0}; //!< Reserved bytes.
    uint16_t m_reserved1{0}; //!< Reserved bytes.
    ScannerInfoVector m_scannerInfos{}; //!< Vector of scannerInfos.
    ScanPointVector m_scanPoints{}; //!< Vector of scanPoints.
}; // Scan2205Container

//==============================================================================

bool operator==(const Scan2205& lhs, const Scan2205& rhs);
bool operator!=(const Scan2205& lhs, const Scan2205& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

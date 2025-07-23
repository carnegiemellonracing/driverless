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

#include <microvision/common/sdk/datablocks/scan/special/ScannerInfoIn2209.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScanPointIn2209.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Same as 0x2205 but uint32 for serialized number of scanpoints.
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2209 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ScannerInfoVector = std::vector<ScannerInfoIn2209>;
    using ScanPointVector   = std::vector<ScanPointIn2209>;

public:
    //! Mask for all processing status flags ("...labeled").
    static constexpr uint32_t labelFlagMask{0xFFFFFFE0U};

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
        VehicleCoordinates
        = 0x00000800U, //!< Bit 11: scan point coordinate system: 0 = scanner coordinates, 1 = vehicle / reference coordinates
        ObjIdAsSegId      = 0x00001000U, //!< Bit 12: segment ID of scan points is used as object ID
        RayAngleCorrected = 0x00002000U, //!< Bit 13: ray angle corrected (window refraction, etc.)
        // 0x00004000 reserved for rotation order
        // 0x00008000 reserved for rotation order
        // 0x00010000 reserved for rotation order
        HorizontalIdAsSegId = 0x00020000U //!< Bit 17: scan points segmentId contains horizontal id for MOVIA scan
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2209"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Scan2209();
    ~Scan2209() override = default;

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

    //! Returns true if the scan points are given in vehicle coordinates, or false if the scan points are given in the Laserscanner coordinate system.
    bool isVehicleCoordinates() const { return isFlagSet(Flags::VehicleCoordinates); }

    //! Returns true if the scan points' segment ID is used as object ID.
    bool isSegIDUsedAsObjID() const { return isFlagSet(Flags::ObjIdAsSegId); }

    //! Returns true if the ray angle has been corrected (Window refraction compensation, etc.).
    bool isRayAngleCorrected() const { return isFlagSet(Flags::RayAngleCorrected); }

    //========================================
    //! \brief Returns true if the scan points have been imported from a Scan2340 (MOVIA scan) and the segment id contains the horizontal id.
    //!
    //! \return \c True if scan is imported from Scan2340, \c false otherwise.
    //----------------------------------------
    bool isSegIdUsedAsHorizontalId() const { return isFlagSet(Flags::HorizontalIdAsSegId); }

    uint16_t getScanNumber() const { return m_scanNumber; }

    uint8_t getNumberOfScannerInfos() const { return uint8_t(m_scannerInfos.size()); }

    const ScannerInfoVector& getScannerInfos() const { return m_scannerInfos; }
    ScannerInfoVector& getScannerInfos() { return m_scannerInfos; }

    uint32_t getNumberOfScanPoints() const { return uint32_t(m_scanPoints.size()); }

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
    void setGroundLabeled(const bool groundLabeled = true) { setFlag(Flags::GroundLabeled, groundLabeled); }

    //! Set the Scan has dirt labeled.
    void setDirtLabeled(const bool dirtLabeled = true) { setFlag(Flags::DirtLabeled, dirtLabeled); }

    //! Set the Scan has rain labeled.
    void setRainLabeled(const bool rainLabeled = true) { setFlag(Flags::RainLabeled, rainLabeled); }

    //! Set the Scan has coverage labeled.
    void setCoverageLabeled(const bool coverageLabeled = true) { setFlag(Flags::CoverageLabeled, coverageLabeled); }

    //! Set the Scan has background labeled.
    void setBackgroundLabeled(const bool backgroundLabeled = true)
    {
        setFlag(Flags::BackgroundLabeled, backgroundLabeled);
    }

    //! Set the Scan has reflector labeled.
    void setReflectorLabeled(const bool reflectorLabeled = true) { setFlag(Flags::ReflectorLabeled, reflectorLabeled); }

    //! Set the flag which says this scan originated from an upside down mounted scanner.
    void setUpsideDown(const bool upsideDown = true) { setFlag(Flags::UpsideDown, upsideDown); }

    //! Set whether the scan was calculated from the fusion of several scans
    void setFusionScan(const bool fusionScan = true) { setFlag(Flags::Fusion, fusionScan); }

    //! Set whether the scanpoints are given in vehicle coordinates
    void setVehicleCoordinates(const bool vehicleCoordinates = true)
    {
        setFlag(Flags::VehicleCoordinates, vehicleCoordinates);
    }

    //! Set whether the scan points' segment ID is used as object ID.
    void setSegIDUsedAsObjID(const bool segIdUsedAsObjId = true) { setFlag(Flags::ObjIdAsSegId, segIdUsedAsObjId); }

    //! Set whether the ray angle has been corrected (Window refraction compensation, etc.)
    void setRayAngleCorrected(const bool rayAngleCorrected = true)
    {
        setFlag(Flags::RayAngleCorrected, rayAngleCorrected);
    }

    //========================================
    //! \brief Set whether the segment id of the scan points contains the horizontal id from a MOVIA scan.
    //!
    //! \param[in] segmentIdIshorizontalId  \c True if scan is imported from Scan2340, \c false otherwise.
    //----------------------------------------
    void setSegIdUsedAsHorizontalId(const bool segmentIdIshorizontalId = true)
    {
        setFlag(Flags::HorizontalIdAsSegId, segmentIdIshorizontalId);
    }

    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }

    void setReserved0(const uint8_t newReserved0) { m_reserved0 = newReserved0; }
    void setReserved1(const uint16_t newReserved1) { m_reserved1 = newReserved1; }

    void setScannerInfos(const ScannerInfoVector& newScannerInfos) { m_scannerInfos = newScannerInfos; }
    void setScannerInfos(ScannerInfoVector&& newScannerInfos) { m_scannerInfos = std::move(newScannerInfos); }

    void setScanPoints(const ScanPointVector& newScanPoints) { m_scanPoints = newScanPoints; }
    void setScanPoints(ScanPointVector&& newScanPoints) { m_scanPoints = std::move(newScanPoints); }

protected:
    NtpTime m_scanStartTime{};
    uint32_t m_endTimeOffset{};
    uint32_t m_flags{0};
    uint16_t m_scanNumber{0};
    //	uint32_t number of scan points;
    //	uint8_t number of scan infos;
    uint8_t m_reserved0{0};
    uint16_t m_reserved1{0};
    ScannerInfoVector m_scannerInfos{};
    ScanPointVector m_scanPoints{};
}; // Scan2209Container

//==============================================================================

bool operator==(const Scan2209& lhs, const Scan2209& rhs);
bool operator!=(const Scan2209& lhs, const Scan2209& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

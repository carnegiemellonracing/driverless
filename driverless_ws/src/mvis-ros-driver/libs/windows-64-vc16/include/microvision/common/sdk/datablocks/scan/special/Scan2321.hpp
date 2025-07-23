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
//! \date Jun 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/scan/special/CorrectionSetIn2321.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/ScannerInfoIn2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/SubScanIn2321.hpp>
#include <microvision/common/sdk/datablocks/scan/special/CalibrationDataIn2321.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Container for third party lidar raw data scans (data type 2321).
//!
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2321 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    friend bool operator==(const Scan2321& lhs, const Scan2321& rhs);

public:
    using SubScanVector = std::vector<SubScanIn2321>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2321"};

    //========================================
    //! \brief Get the hash for this container type (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //!
    //! Creates an empty scan.
    //----------------------------------------
    Scan2321() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~Scan2321() = default;

public:
    //========================================
    //! \brief Get the hash for this container type.
    //!
    //! \return hash value for this container type
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the timestamp of this scan including GPS information as received from the scanner.
    //!
    //! \return point in time when the first laser of the first sub-scan is fired.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the consecutive number of this scan.
    //!
    //! \return the number of this scan.
    //----------------------------------------
    uint32_t getScanNumber() const { return m_scanNumber; }

    //========================================
    //! \brief Get the scanner return mode, i.e. whether the scanner returns one (single) or two (dual) points per
    //! laser.
    //!
    //! \return the scanner return mode.
    //----------------------------------------
    ThirdPartyVLidarProtocol::ReturnMode getReturnMode() const { return m_returnMode; }

    //========================================
    //! \brief Get parameter about the scanner.
    //!
    //! \return parameter about the scanner.
    //----------------------------------------
    const ScannerInfoIn2321& getScannerInfo() const { return m_scannerInfo; }

    //========================================
    //! \brief Get data used to adjust the coordinates according to the calibration.
    //!
    //! \return the calibration data.
    //----------------------------------------
    CalibrationDataIn2321 getCalibrationData() const { return m_calibrationData; }

    //========================================
    //! \brief Get the number of sub-scans in this scan.
    //!
    //! \return the number of sub-scans.
    //----------------------------------------
    uint16_t getNbOfSubScans() const { return static_cast<uint16_t>(m_subScans.size()); }

    //========================================
    //! \brief Get the sub-scans in this scan.
    //!
    //! \return the sub-scans.
    //----------------------------------------
    SubScanVector getSubScans() const { return m_subScans; }

public: // setter
    //========================================
    //! \brief Set the timestamp including GPS information as received from the scanner.
    //!
    //! \param[in] timestamp  the new timestamp.
    //----------------------------------------
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }

    //========================================
    //! \brief Set the consecutive number of this scan.
    //!
    //! \param[in] scanNumber  the new scan number
    //----------------------------------------
    void setScanNumber(const uint32_t scanNumber) { m_scanNumber = scanNumber; }

    //========================================
    //! \brief Set the scanner return mode, i.e. whether the scanner returns one (single) or two (dual) points per
    //! laser.
    //!
    //! \param[in] returnMode  the new return mode
    //----------------------------------------
    void setReturnMode(const ThirdPartyVLidarProtocol::ReturnMode returnMode) { m_returnMode = returnMode; }

    //========================================
    //! \brief Set parameter about the scanner.
    //!
    //! \param[in] scannerInfo  the new scanner information
    //----------------------------------------
    void setScannerInfo(const ScannerInfoIn2321& scannerInfo) { m_scannerInfo = scannerInfo; }

    //========================================
    //! \brief Set data used to adjust the coordinates according to the calibration.
    //!
    //! \param[in] calibrationData  the new calibration data
    //----------------------------------------
    void setCalibrationData(const CalibrationDataIn2321& calibrationData) { m_calibrationData = calibrationData; }

    //========================================
    //! \brief Set the sub-scans in this scan.
    //!
    //! \param[in] subScans  the new sub-scans
    //----------------------------------------
    void setSubScans(const SubScanVector& subScans) { m_subScans = subScans; }

protected:
    NtpTime m_timestamp{0}; //!< The time when the first measurement was done.
    uint32_t m_scanNumber{std::numeric_limits<uint32_t>::max()}; //!< The number of this scan.
    //! The return mode the laser is operating in.
    ThirdPartyVLidarProtocol::ReturnMode m_returnMode{ThirdPartyVLidarProtocol::ReturnMode::Unknown};
    ScannerInfoIn2321 m_scannerInfo; //!< Additional scanner information in the format of the 2205 scanner information.
    CalibrationDataIn2321 m_calibrationData; //!< Individual parameters measured during calibration of this  scanner.
    SubScanVector m_subScans; //!< Vector of sub-scans.
}; // Scan2321

//==============================================================================

//==============================================================================
//! \brief Test scans for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise
//------------------------------------------------------------------------------
bool operator==(const Scan2321& lhs, const Scan2321& rhs);

//==============================================================================
//! \brief Test scans for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise
//------------------------------------------------------------------------------
inline bool operator!=(const Scan2321& lhs, const Scan2321& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

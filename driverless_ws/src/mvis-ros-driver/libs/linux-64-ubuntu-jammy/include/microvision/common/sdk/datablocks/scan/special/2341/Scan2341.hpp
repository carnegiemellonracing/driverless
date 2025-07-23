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
//! \date Jun 24, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PerceptionDataInfo.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/ScanPointRowIn2341.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2341/ScannerInfoIn2341.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Container for processed low bandwidth MOVIA scan s.
//!
//! \note The raw data of the MOVIA scan can accessed via datatype \ref microvision::common::sdk::LdmiRaw2352.
//! The high bandwidth MOVIA scan can accessed via datatype \ref microvision::common::sdk::Scan2340.
//------------------------------------------------------------------------------
class MICROVISION_SDK_DEPRECATED Scan2341 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfRows{80}; //!< The size of the row array in a scan.

public:
    using RowArray = std::array<ScanPointRowIn2341, nbOfRows>;

public:
    //========================================
    //! \brief Unique (string) identifier of this class.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2341"};

    //========================================
    //! \brief Get the hash for this container type (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default Constructor.
    //!
    //! Creates an empty scan.
    //----------------------------------------
    Scan2341() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~Scan2341() override = default;

public:
    //========================================
    //! \brief Get the hash for this container type.
    //!
    //! \return The hash value for this container type.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the data info of this scan.
    //!
    //! \return The data info of this scan.
    //----------------------------------------
    const PerceptionDataInfo& getDataInfo() const { return m_dataInfo; }

    //========================================
    //! \brief Get the vector of rows of scanPoints.
    //!
    //! \return The rows of this ScanPoints.
    //----------------------------------------
    const RowArray& getRows() const { return m_rows; }

    //========================================
    //! \brief Get the vector of rows of scanPoints.
    //!
    //! \return The rows of this ScanPoints.
    //----------------------------------------
    RowArray& getRows() { return m_rows; }

    //========================================
    //! \brief Get the value for the cyclic redundancy check.
    //!
    //! \return The check value.
    //----------------------------------------
    uint32_t getCyclicRedundancyCheckValue() const { return m_crc; }

    //========================================
    //! \brief Checks if this scan has a scanner info..
    //!
    //! \return \c True if this scan has a scanner info, \c false otherwise.
    //----------------------------------------
    bool hasScannerInfo() const { return m_hasScannerInfo; }

    //========================================
    //! \brief Get the scannerInfo.
    //!
    //! \return The scanner info.
    //----------------------------------------
    const ScannerInfoIn2341& getScannerInfo() const { return m_scannerInfo; }

public: // setter
    //========================================
    //! \brief Set the data info of this scan.
    //!
    //! \param[in] dataInfo  The new data info.
    //----------------------------------------
    void setDataInfo(const PerceptionDataInfo& dataInfo) { m_dataInfo = dataInfo; }

    //========================================
    //! \brief Set the rows of scanPoints.
    //!
    //! \param[in] rows  The new rows of scanPoints.
    //----------------------------------------
    void setRows(const RowArray& rows) { m_rows = rows; }

    //========================================
    //! \brief Set the value for the cyclic redundancy check.
    //!
    //! \param[in] value  The check value.
    //----------------------------------------
    void setCyclicRedundancyCheckValue(const uint32_t crc) { m_crc = crc; }

    //========================================
    //! \brief Set the scanner info flag.
    //!
    //! \param[in] flag  The new flag indicating if scanner infos are set.
    //----------------------------------------
    void setHasScannerInfo(const bool flag) { m_hasScannerInfo = flag; }

    //========================================
    //! \brief Set the scannerInfo.
    //!
    //! \param[in] scannerInfo  The new scannerInfo.
    //----------------------------------------
    void setScannerInfo(const ScannerInfoIn2341& scannerInfo) { m_scannerInfo = scannerInfo; }

private:
    PerceptionDataInfo m_dataInfo{}; //!< The perception API data header.
    RowArray m_rows{}; //!< The array of Rows of scanPoints.
    uint32_t m_crc{0}; //!< The cyclic redundancy check value over header and row.
    bool m_hasScannerInfo{false}; //!< A flag that indicates if the scan has a scanner info.

    //========================================
    //! Optional scanner info for transformation.
    //!
    //! E.g. For transformation in \sa microvision::common::sdk::Scan2340
    //! \note The scanner info is only serialized if the corresponding flag is active.
    //----------------------------------------
    ScannerInfoIn2341 m_scannerInfo{};
}; // Scan2341

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const Scan2341& lhs, const Scan2341& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Scan2341& lhs, const Scan2341& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
//!\file
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

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PerceptionDataInfo.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/ScanPointRowIn2342.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/ScannerInfoIn2342.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/misc/unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Container for processed low bandwidth MOVIA scan with blockage information.
//!
//! \note The raw data of the MOVIA scan can accessed via datatype \ref microvision::common::sdk::LdmiRaw2352.
//! The high bandwidth MOVIA scan can accessed via datatype \ref microvision::common::sdk::Scan2340.
//------------------------------------------------------------------------------
class Scan2342 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static constexpr uint8_t nbOfRows{80}; //!< The size of the row array in a scan.

    using DegToRad = unit::Convert<unit::angle::degree, unit::angle::radian, double>; //!< Converter for Deg to Rad
    static constexpr double nominalHFovFor11deg
        = Scan2342::DegToRad()(11.00); //!< Nominal horizontal FoV for 11 deg sensors
    static constexpr double nominalVFovFor11deg
        = Scan2342::DegToRad()(18.75); //!< Nominal vertical FoV for 11 deg sensors
    static constexpr double nominalHFovFor60deg
        = Scan2342::DegToRad()(60.00); //!< Nominal horizontal FoV for 60 deg sensors
    static constexpr double nominalVFovFor60deg
        = Scan2342::DegToRad()(37.50); //!< Nominal vertical FoV for 60 deg sensors
    static constexpr double nominalHFovFor120deg
        = Scan2342::DegToRad()(120.00); //!< Nominal horizontal FoV for 120 deg sensors
    static constexpr double nominalVFovFor120deg
        = Scan2342::DegToRad()(75.00); //!< Nominal vertical FoV for 120 deg sensors

public:
    using RowArray = std::array<ScanPointRowIn2342, nbOfRows>;

public:
    //========================================
    //! \brief Status to identify if the sensor is blocked.
    //----------------------------------------
    enum class Blockage : uint8_t
    {
        NotAvailable = 0, //!< The blockage information is not available.
        Clean        = 1, //!< Sensor is considered as clean.
        Blocked      = 255, //!< Sensor is considered as blocked.
    };

    //========================================
    //! \brief Status to identify the detection range of the sensor.
    //----------------------------------------
    enum class RangeClassifier : uint8_t
    {
        NotAvailable       = 0, //!< The detection range information not available.
        LowPerformance     = 1, //!< The detection range is considered as low.
        MediumPerformance  = 128, //!< The detection range is considered as medium.
        MaximumPerformance = 255, //!< The detection range is considered as high.
    };

    //========================================
    //! \brief Status to identify the echo sorting of the sensor.
    //----------------------------------------
    enum class EchoSortingClassifier : uint8_t
    {
        None                   = 0x00U, //!< Arbitrary ordering.
        HighestIntensityFirst  = 0x01U, //!< Highest intensity first.
        SmallestIntensityFirst = 0x02U, //!< Smallest intensity first.
        NearestDistanceFirst   = 0x03U, //!< Nearest (smallest) distance first.
        FarthestDistanceFirst  = 0x04U, //!< Farthest (largest) distance first.
    };

public:
    //========================================
    //! \brief Unique (string) identifier of this class.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2342"};

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
    Scan2342() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~Scan2342() override = default;

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
    //! \brief Get the blockage state of the scan.
    //!
    //! \return The blockage state.
    //----------------------------------------
    Blockage getBlockage() const { return m_blockage; }

    //========================================
    //! \brief Get the range classifier of the scan.
    //!
    //! \return The range classifier.
    //----------------------------------------
    RangeClassifier getRange() const { return m_range; }

    //========================================
    //! \brief Get the echo sorting classifier of the scan.
    //!
    //! \return The echo sorting classifier.
    //----------------------------------------
    EchoSortingClassifier getEchoSorting() const { return m_echoSorting; }

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
    const ScannerInfoIn2342& getScannerInfo() const { return m_scannerInfo; }

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
    //! \brief Set the blockage state.
    //!
    //! \param[in] blockage  The new blockage state of the scan.
    //----------------------------------------
    void setBlockage(const Blockage blockage) { m_blockage = blockage; }

    //========================================
    //! \brief Set the range classifier.
    //!
    //! \param[in] range  The new range classifier of the scan.
    //----------------------------------------
    void setRange(const RangeClassifier range) { m_range = range; }

    //========================================
    //! \brief Set the echo sorting classifier.
    //!
    //! \param[in] echoSorting  The new echo sorting classifier of the scan.
    //----------------------------------------
    void setEchoSorting(const EchoSortingClassifier echoSorting) { m_echoSorting = echoSorting; }

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
    void setScannerInfo(const ScannerInfoIn2342& scannerInfo) { m_scannerInfo = scannerInfo; }

private:
    PerceptionDataInfo m_dataInfo{}; //!< The perception API data header.
    RowArray m_rows{}; //!< The array of Rows of scanPoints.
    Blockage m_blockage{Blockage::NotAvailable}; //!< State of blockage for the sensor.
    RangeClassifier m_range{
        RangeClassifier::NotAvailable}; //!< Status of the detection range of the sensor. Now reserved byte 1.
    uint8_t m_reserved2{}; //!< Reserved byte.
    EchoSortingClassifier m_echoSorting{EchoSortingClassifier::None}; //!< Sorting of echoes in pixels.
    uint32_t m_crc{0}; //!< The cyclic redundancy check value over header and row.
    bool m_hasScannerInfo{false}; //!< A flag that indicates if the scan has a scanner info.

    //========================================
    //! Optional scanner info for transformation.
    //!
    //! E.g. For transformation in \sa microvision::common::sdk::Scan2340
    //! \note The scanner info is only serialized if the corresponding flag is active.
    //----------------------------------------
    ScannerInfoIn2342 m_scannerInfo{};
}; // Scan2342

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const Scan2342& lhs, const Scan2342& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Scan2342& lhs, const Scan2342& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

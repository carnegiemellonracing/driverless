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
//! \date Nov 21, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/ScannerInfoIn2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/ScanPointIn2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2340/ScanPointInfoListIn2340.hpp>
#include <microvision/common/sdk/datablocks/scan/special/2342/Scan2342.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Container for processed MOVIA scan s.
//!
//! \note The raw data of the MOVIA scan can accessed via datatype 2352.
//! General data type: \ref microvision::common::sdk::Scan
//------------------------------------------------------------------------------
class Scan2340 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ScanPointVector         = std::vector<ScanPointIn2340>;
    using ScanPointInfoListVector = std::vector<ScanPointInfoListIn2340>;

public:
    //========================================
    //! \brief Unique (string) identifier of this class.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.scan2340"};

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
    Scan2340() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~Scan2340() override = default;

public:
    //========================================
    //! \brief Get the hash for this container type.
    //!
    //! \return The hash value for this container type
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the start timestamp of this scan as received from the scanner.
    //!
    //! \return Point in time when the laser of the first scan point is fired.
    //----------------------------------------
    NtpTime getScanStartTimestamp() const { return m_scanStartTime; }

    //========================================
    //! \brief Get the end timestamp of this scan as received from the scanner.
    //!
    //! \return Point in time when the laser of the last scan point is fired.
    //----------------------------------------
    NtpTime getScanEndTimestamp() const { return m_scanEndTime; }

    //========================================
    //! \brief Get scanner setup information, e.g. like mounting position, field of view...
    //!
    //! \return  The scanner information.
    //----------------------------------------
    const ScannerInfoIn2340& getScannerInfo() const { return m_scannerInfo; }

    //========================================
    //! \brief Get the sequential id of this scan.
    //!
    //! \return The id of this scan.
    //! \note There will be an overflow after 2^32 ids.
    //----------------------------------------
    uint32_t getScanNumber() const { return m_scanNumber; }

    //========================================
    //! \brief Get the scan's scan points.
    //!
    //! The points are sorted row-major (vertical-major) and echo-minor, i.e.
    //! 1. points on the same beam (verticalId and horizontalId identically) stored
    //!    consecutively (ordered by ascending echo id)
    //! 2. all points of a row stored consecutively (ordered by ascending
    //!    horizontal id).
    //! 3. The rows (points with identical verticalId) will be stored ordered
    //!    by ascending verticalId.
    //!
    //! \note The scan points are sorted in row-major format:
    //! \code{.unparsed}
    //!       vertical_id
    //!       ^
    //!       |  8    9   10   11
    //!       |  4    5    6    7
    //!       |  0    1    2    3
    //!       +-------------------->
    //!                    horizontal_id
    //! \endcode
    //! All echoes for a (vertical_id, horizontal_id) pair are placed consecutively (echo-minor)
    //! in the point's vector. Example, tuples (vertical_id, horizontal_id, echo_id) given:
    //! \code{.unparsed}
    //!       (0, 0, 0)
    //!       (0, 0, 1)
    //!       (0, 0, 2)
    //!       (0, 1, 0)
    //!       (0, 1, 1)
    //!       (0, 2, 0)
    //!       (1, 0, 0) ...
    //! \endcode
    //!
    //! \return The scan points in the described order.
    //----------------------------------------
    const ScanPointVector& getScanPoints() const { return m_scanPoints; }

    //========================================
    //! \brief Get optional parameters about the scan point.
    //!
    //! The returned ScanPointInfoListVector contains zero or more
    //! ScanPointInfoListIn2340. Each of them is containing a vector
    //! with an additional value for each scan point in this scan.
    //!
    //! Each ScanPointInfoListIn2340 has a ScanPointInfoListIn2340::InformationType
    //! describing the meaning of the float value.
    //!
    //! ScanPointInfoListVector shall have not more than one ScanPointInfoListIn2340
    //! per type.
    //!
    //! Combining the scan point and the optional scan point information, the same
    //! vector index has to be used for the scan point and the optional value in their
    //! vectors.
    //!
    //! \return A vector of optional scan point values.
    //----------------------------------------
    const ScanPointInfoListVector& getScanPointInfos() const { return m_scanPointInfos; }

    //========================================
    //! \brief Get the blockage state of the scan.
    //!
    //! \return \c True if the sensor is blocked, \c false if clean.
    //----------------------------------------
    bool isBlocked() const { return (m_blockage == Scan2342::Blockage::Blocked); }

    //========================================
    //! \brief Get the blockage state of the scan.
    //!
    //! \return The blockage state of the scan.
    //----------------------------------------
    Scan2342::Blockage getBlockage() const { return m_blockage; }

    //========================================
    //! \brief Get the range classifier of the scan.
    //!
    //! \return The range classifier of the scan.
    //----------------------------------------
    Scan2342::RangeClassifier getRange() const { return m_range; }

public: // setter
    //========================================
    //! \brief Set the start timestamp of this scan.
    //!
    //! \param[in] timestamp  The new point of time when the laser of the first scan point is fired.
    //----------------------------------------
    void setScanStartTimestamp(const NtpTime timestamp) { m_scanStartTime = timestamp; }

    //========================================
    //! \brief Set the end timestamp of this scan.
    //!
    //! \param[in] timestamp  The new point of time when the laser of the last scan point is fired.
    //----------------------------------------
    void setScanEndTimestamp(const NtpTime timestamp) { m_scanEndTime = timestamp; }

    //========================================
    //! \brief Set scanner setup information.
    //!
    //! \param[in] scannerInfo  The new scanner setup information.
    //----------------------------------------
    void setScannerInfo(const ScannerInfoIn2340& scannerInfo) { m_scannerInfo = scannerInfo; }

    //========================================
    //! \brief Set the sequential id of this scan.
    //!
    //! \param[in] scanNumber  The new scan id.
    //----------------------------------------
    void setScanNumber(const uint32_t scanNumber) { m_scanNumber = scanNumber; }

    //========================================
    //! \brief Set the scan's scan points.
    //!
    //! The scan points are sorted in the order described in getScanPoint.
    //!
    //! \param[in] scanPoints  The new vector of scan points in the described order.
    //!
    //! \sa getScanPoints
    //----------------------------------------
    void setScanPoints(const ScanPointVector& scanPoints) { m_scanPoints = scanPoints; }

    //========================================
    //! \brief Move \a scanPoints into scan.
    //!
    //! The scan points are sorted in the order described in getScanPoint.
    //!
    //! \param[in] scanPoints  The new vector of scan points in the described order.
    //!
    //! \sa getScanPoints, setScanPoints(const ScanPointVector&)
    //----------------------------------------
    void setScanPoints(ScanPointVector&& scanPoints) { m_scanPoints = std::move(scanPoints); }

    //========================================
    //! \brief Set optional information of the scan points.
    //!
    //! \a scanPointInfos contains zero or more
    //! ScanPointInfoListIn2340. Each of them is containing a vector
    //! with an additional value for each scan point in this scan.
    //!
    //! Each ScanPointInfoListIn2340 has a ScanPointInfoListIn2340::InformationType
    //! describing the meaning of the float value.
    //!
    //! ScanPointInfoListVector shall have not more than one ScanPointInfoListIn2340
    //! per type.
    //!
    //! Combining the scan point and the optional scan point information, the same
    //! vector index has to be used for the scan point and the optional value in their
    //! vectors.
    //!
    //! \param[in] scanPointInfos  The new scan point information.
    //----------------------------------------
    void setScanPointInfos(const ScanPointInfoListVector& scanPointInfos) { m_scanPointInfos = scanPointInfos; }

    //========================================
    //! \brief Move optional information of scan points into the scan.
    //!
    //! \param[in] scanPointInfos  The new scan point information.
    //! \sa setScanPointInfos(const ScanPointInfoListVector&)
    //----------------------------------------
    void setScanPointInfos(ScanPointInfoListVector&& scanPointInfos) { m_scanPointInfos = std::move(scanPointInfos); }

private:
    NtpTime m_scanStartTime{}; //!< Timestamp of the first scan point in this scan.
    NtpTime m_scanEndTime{}; //!< Timestamp of the last scan point in this scan.
    ScannerInfoIn2340 m_scannerInfo{}; //!< The parameter about the scanner (mounting position, field of view, etc.).
    uint32_t m_scanNumber{0U}; //!< Sequential id of this scan.
    ScanPointVector m_scanPoints{}; //!< Ordered scan points of this scan. The order is described in getScanPoint.
    ScanPointInfoListVector m_scanPointInfos{}; //!< Vector with optional measured properties of each scan point.

    //!@{
    //!This members are not serialized!
    Scan2342::Blockage m_blockage{Scan2342::Blockage::NotAvailable}; //!< State of blockage for the sensor.
    Scan2342::RangeClassifier m_range{Scan2342::RangeClassifier::NotAvailable}; //!< Range classifier for the sensor.
    //!@}
}; // Scan2340

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const Scan2340& lhs, const Scan2340& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Scan2340& lhs, const Scan2340& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

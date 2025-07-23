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
#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelIn6503.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of object labels
//!
//! General data type: \ref microvision::common::sdk::ObjectLabelList
//------------------------------------------------------------------------------
class ObjectLabelList6503 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const uint32_t nbOfReserved{10};

public: //type definitions
    using LabelVector   = std::vector<ObjectLabelIn6503>;
    using ReservedArray = std::array<uint16_t, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlabellist6503"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectLabelList6503() : SpecializedDataContainer() {}
    virtual ~ObjectLabelList6503() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    uint32_t getTimeOffsetUs() const { return m_timeOffsetUs; }
    NtpTime getTimestamp() const { return m_timestamp; }
    uint32_t getScanNumber() const { return m_scanNumber; }
    NtpTime getScanMidTimestamp() const { return m_scanMidTimestamp; }
    uint16_t getReserved(const uint32_t idx) const { return m_reserved.at(idx); }

    const LabelVector& getLabels() const { return m_labels; }
    LabelVector& getLabels() { return m_labels; }

public: // setter
    void setTimeOffsetUs(const uint32_t newTimeOffsetUs) { m_timeOffsetUs = newTimeOffsetUs; }
    void setTimestamp(const NtpTime newTimestamp) { m_timestamp = newTimestamp; }
    void setScanNumber(const uint32_t newScanNumber) { m_scanNumber = newScanNumber; }
    void setScanMidTimestamp(const NtpTime newScanMidTimestamp) { m_scanMidTimestamp = newScanMidTimestamp; }
    void setReserved(const uint32_t idx, const uint16_t newReservedValue) { m_reserved.at(idx) = newReservedValue; }
    // use getLabels

protected:
    uint32_t m_timeOffsetUs{0}; //!< The time offset in microseconds.
    NtpTime m_timestamp{0}; //!< The timestamp of the object label list.
    uint32_t m_scanNumber{0}; //!< The corresponding scan number.
    NtpTime m_scanMidTimestamp{0}; //!< The mid scan timestamp.
    ReservedArray m_reserved{{}}; //!< Reserved bytes.
    //uint16_t m_nbOfLabels
    LabelVector m_labels{}; //!< The vector of object labels.
}; // ObjectLabelList6503

//==============================================================================

bool operator==(const ObjectLabelList6503& lhs, const ObjectLabelList6503& rhs);
bool operator!=(const ObjectLabelList6503& lhs, const ObjectLabelList6503& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

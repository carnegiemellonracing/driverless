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
//! \date Jan 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/ObjectLabel.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelList6503.hpp>

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
//! Special data type: \ref microvision::common::sdk::ObjectLabelList6503
//------------------------------------------------------------------------------
class ObjectLabelList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const ObjectLabelList& lhs, const ObjectLabelList& rhs);

public:
    constexpr static const uint32_t nbOfReserved{10};

public: //type definitions
    using LabelVector   = std::vector<ObjectLabel>;
    using ReservedArray = std::array<uint16_t, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.objectlabellist"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectLabelList() : DataContainerBase() {}
    virtual ~ObjectLabelList() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    uint32_t getTimeOffsetUs() const { return m_delegate.getTimeOffsetUs(); }
    NtpTime getTimestamp() const { return m_delegate.getTimestamp(); }
    uint32_t getScanNumber() const { return m_delegate.getScanNumber(); }
    NtpTime getScanMidTimestamp() const { return m_delegate.getScanMidTimestamp(); }
    uint16_t getReserved(const uint32_t idx) const { return m_delegate.getReserved(idx); }

    const LabelVector& getLabels() const { return m_delegate.getLabels(); }
    LabelVector& getLabels() { return m_delegate.getLabels(); }

public: // setter
    void setTimeOffsetUs(const uint32_t newTimeOffsetUs) { m_delegate.setTimeOffsetUs(newTimeOffsetUs); }
    void setTimestamp(const NtpTime newTimestamp) { m_delegate.setTimestamp(newTimestamp); }
    void setScanNumber(const uint32_t newScanNumber) { m_delegate.setScanNumber(newScanNumber); }
    void setScanMidTimestamp(const NtpTime newScanMidTimestamp) { m_delegate.setScanMidTimestamp(newScanMidTimestamp); }
    void setReserved(const uint32_t idx, const uint16_t newReservedValue)
    {
        m_delegate.setReserved(idx, newReservedValue);
    }
    // use getLabels

private:
    ObjectLabelList6503 m_delegate;
}; // ObjectLabelList

//==============================================================================

inline bool operator==(const ObjectLabelList& lhs, const ObjectLabelList& rhs)
{
    return lhs.m_delegate == rhs.m_delegate;
}

inline bool operator!=(const ObjectLabelList& lhs, const ObjectLabelList& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

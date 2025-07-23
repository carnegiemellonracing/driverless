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
//! \date Mar 25th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821.hpp>

#include <microvision/common/sdk/datablocks/MeasurementSeries.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief MeasurementSeries list of measurement snippets
//!
//! Special data type: \ref microvision::common::sdk::MeasurementList2821
//------------------------------------------------------------------------------
class MeasurementList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const MeasurementList& lhs, const MeasurementList& rhs);

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.measurementlist"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    MeasurementList();
    ~MeasurementList() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint32_t getMicroseconds() const { return m_delegate.getMicroseconds(); }
    NtpTime getTimestamp() const { return m_delegate.getTimestamp(); }

    const std::string& getListName() const { return m_delegate.getListName(); }
    std::string& getListName() { return m_delegate.getListName(); }
    const std::string& getGroupName() const { return m_delegate.getGroupName(); }
    std::string& getGroupName() { return m_delegate.getGroupName(); }
    const MeasurementSeries& getMeasurements() const { return m_delegate.getMeasList(); }
    MeasurementSeries& getMeasurements() { return m_delegate.getMeasList(); }

public:
    void setMicroseconds(const uint32_t newMicroseconds) { m_delegate.setMicroseconds(newMicroseconds); }
    void setTimestamp(const NtpTime newTimestamp) { m_delegate.setTimestamp(newTimestamp); }
    void setListName(const std::string& newListName) { m_delegate.setListName(newListName.c_str()); }
    void setGroupName(const std::string& newGroupName) { m_delegate.setGroupName(newGroupName.c_str()); }
    void setMeasurements(const MeasurementSeries& newMeasurementList)
    {
        m_delegate.setMeasurements(newMeasurementList);
    }

protected:
    MeasurementList2821 m_delegate; // only possible specialization currently
}; // MeasurementListContainer

//==============================================================================

bool operator==(const MeasurementList& lhs, const MeasurementList& rhs);
bool operator!=(const MeasurementList& lhs, const MeasurementList& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

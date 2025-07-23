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
//! \date March 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9010.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of relations for the reference timebase to an
//! 'Other' timebase
//!
//! General data type: \ref microvision::common::sdk::TimeRelationsList
//------------------------------------------------------------------------------
class TimeRelationsList9010 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using TimeRelationMap = std::map<ClockType, TimeRelationIn9010>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.timerelations9010"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TimeRelationsList9010();
    virtual ~TimeRelationsList9010();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // Setter
    void setTimeRelations(const TimeRelationMap& timerRelations) { m_timeRelationsMap = timerRelations; }

public: // Getter
    const TimeRelationMap& getTimeRelations() const { return m_timeRelationsMap; }

    // Wrapper for getTimeRelationsMap
    TimeRelationMap& getTimeRelations() { return m_timeRelationsMap; }

protected:
    TimeRelationMap m_timeRelationsMap;
}; // TimeRelationsList9010

//==============================================================================

bool operator==(const TimeRelationsList9010& lhs, const TimeRelationsList9010& rhs);

//==============================================================================

bool operator!=(const TimeRelationsList9010& lhs, const TimeRelationsList9010& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

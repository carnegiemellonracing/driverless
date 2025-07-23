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
#include <microvision/common/sdk/datablocks/ClockType.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationIn9011.hpp>
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
class TimeRelationsList9011 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using TimeRelationVector = std::vector<TimeRelationIn9011>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.timerelationslist9011"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TimeRelationsList9011();
    ~TimeRelationsList9011() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // setter
    void setRefClock(const ClockType refClock) { m_referenceClock = refClock; }

    void setTimeRelations(const TimeRelationVector& timerRelations) { m_timeRelations = timerRelations; }

public: // getter
    const ClockType getRefClock() const { return m_referenceClock; }

    const TimeRelationVector& getTimeRelations() const { return m_timeRelations; }

    // Wrapper for getTimeRelationsMap
    TimeRelationVector& getTimeRelations() { return m_timeRelations; }

protected:
    ClockType m_referenceClock;
    TimeRelationVector m_timeRelations;
}; // TimeRelationsList9011

//==============================================================================

bool operator==(const TimeRelationsList9011& lhs, const TimeRelationsList9011& rhs);

//==============================================================================

bool operator!=(const TimeRelationsList9011& lhs, const TimeRelationsList9011& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

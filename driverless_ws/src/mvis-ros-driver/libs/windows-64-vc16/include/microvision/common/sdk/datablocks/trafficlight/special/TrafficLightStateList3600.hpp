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
//! \date Aug 29, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/trafficlight/special/TrafficLightStateIn3600.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief A container for all traffic lights and their state.
//!
//! General data type: \ref microvision::common::sdk::TrafficLightStateList
//------------------------------------------------------------------------------
class TrafficLightStateList3600 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.trafficlightstatelist3600"};
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    TrafficLightStateList3600();
    ~TrafficLightStateList3600() override = default;

public:
    //! Return list of all traffic light states.
    const std::vector<TrafficLightStateIn3600>& getTrafficLightStates() const { return m_trafficLightStates; }

public:
    //! Set list of traffic light states.
    void setTrafficLightStates(const std::vector<TrafficLightStateIn3600>& trafficLightStates)
    {
        m_trafficLightStates = trafficLightStates;
    }

    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

private:
    std::vector<TrafficLightStateIn3600> m_trafficLightStates{};

}; // TrafficLightStateList

//==============================================================================

bool operator==(const TrafficLightStateList3600& lhs, const TrafficLightStateList3600& rhs);
bool operator!=(const TrafficLightStateList3600& lhs, const TrafficLightStateList3600& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

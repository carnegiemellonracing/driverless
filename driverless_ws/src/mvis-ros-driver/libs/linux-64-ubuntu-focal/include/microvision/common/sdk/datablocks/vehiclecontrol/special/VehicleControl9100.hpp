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
//! \date Mar 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief VehicleControl
//!
//! General data type: \ref microvision::common::sdk::VehicleControl
//------------------------------------------------------------------------------
class VehicleControl9100 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint8_t nbOfReserved = 4;

public:
    using ReservedArray = std::array<uint32_t, nbOfReserved>;

public:
    //========================================
    //!\enum SteeringType
    //! This type indicates if the steering value is dedicated to the
    //! torque or angle interface of a vehicle.
    //----------------------------------------
    enum class SteeringType : uint8_t
    {
        SteeringWheelTorque = 0,
        SteeringWheelAngle  = 1,
        FrontWheelAngle     = 2
    };

    //========================================
    //!\enum IndicatorState
    //! Type for different indicator states
    //----------------------------------------
    enum class IndicatorState : uint8_t
    {
        Off   = 0,
        Left  = 1,
        Right = 2,
        Both  = 3
    };

    //========================================
    //!\enum  SourceType
    //! Possible sources for vehicle control data.
    //----------------------------------------
    enum class SourceType : uint8_t
    {
        Controller          = 0,
        Keyboard            = 1,
        Joystick            = 2,
        AccelrationOverride = 3,
        FallBack            = 4,
        Unknown             = 255
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclecontrol9100"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    VehicleControl9100()          = default;
    virtual ~VehicleControl9100() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    void setTimestamp(const Timestamp& timestamp) { this->m_timestamp = timestamp; }
    void setSourceType(const SourceType sourceType) { this->m_sourceType = sourceType; }
    void setSteeringType(const SteeringType steeringType) { this->m_steeringType = steeringType; }
    void setSteeringValue(const float steeringValue) { this->m_steeringValue = steeringValue; }
    void setIndicatorState(const IndicatorState indicatorState) { this->m_indicatorState = indicatorState; }
    void setAccelerationValue(const float accelerationValue) { this->m_accelerationValue = accelerationValue; }
    void setDoStop(const bool doStop) { this->m_doStop = doStop; }
    void setStopDistance(const float stopDistance) { this->m_stopDistance = stopDistance; }

    const Timestamp& getTimestamp() const { return m_timestamp; }
    SourceType getSourceType() const { return m_sourceType; }
    SteeringType getSteeringType() const { return m_steeringType; }
    float getSteeringValue() const { return m_steeringValue; }
    IndicatorState getIndicatorState() const { return m_indicatorState; }
    float getAccelerationValue() const { return m_accelerationValue; }
    bool isDoStop() const { return m_doStop; }
    float getStopDistance() const { return m_stopDistance; }

protected:
    Timestamp m_timestamp{}; //!< Timestamp of VehicleControl
    SourceType m_sourceType{SourceType::Unknown}; //!< Vehicle control source type
    SteeringType m_steeringType{SteeringType::FrontWheelAngle}; //!< Vehicle control steering type
    float m_steeringValue{NaN}; //!< Steering value for steering wheel torque or angle interface
    IndicatorState m_indicatorState{IndicatorState::Off}; //!< Desired indicator state
    float m_accelerationValue{NaN}; //!< Vehicle Controls Acceleration
    bool m_doStop{false}; //!< stop intention (mostly necessary for low level state machines)
    float m_stopDistance{NaN}; //!< desired stop distance (mostly necessary for low level controllers)

private:
    ReservedArray m_reserved;
}; // VehicleControl9100

//==============================================================================

bool operator==(const VehicleControl9100& lhs, const VehicleControl9100& rhs);
bool operator!=(const VehicleControl9100& lhs, const VehicleControl9100& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

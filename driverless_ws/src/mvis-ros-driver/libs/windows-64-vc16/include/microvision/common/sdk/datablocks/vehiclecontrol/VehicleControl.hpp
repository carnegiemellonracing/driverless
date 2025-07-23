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
//! \date May 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/datablocks/vehiclecontrol/special/VehicleControl9100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief VehicleControl
//!
//! Special data type: \ref microvision::common::sdk::VehicleControl9100
//------------------------------------------------------------------------------
class VehicleControl final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.generalcontainer.vehiclecontrol"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    VehicleControl();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~VehicleControl() = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Set the timestamp of the VehicleControl
    //!\param[in] timestamp  New timestamp of the VehicleControl
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_delegate.setTimestamp(timestamp); }

    //========================================
    //!\brief Set the sourceType of the VehicleControl
    //!\param[in] sourceType  New  sourceType of the VehicleControl
    //----------------------------------------
    void setSourceType(const VehicleControl9100::SourceType sourceType) { m_delegate.setSourceType(sourceType); }

    //========================================
    //!\brief Set the steeringType of the VehicleControl
    //!\param[in] steeringType  New steeringType of the VehicleControl
    //----------------------------------------
    void setSteeringType(const VehicleControl9100::SteeringType steeringType)
    {
        m_delegate.setSteeringType(steeringType);
    }

    //========================================
    //!\brief Set the steeringValue of the VehicleControl
    //!\param[in] steeringValue  New steeringValue of the VehicleControl
    //----------------------------------------
    void setSteeringValue(const float steeringValue) { m_delegate.setSteeringValue(steeringValue); }

    //========================================
    //!\brief Set the  indicatorState of the VehicleControl
    //!\param[in] indicatorState  New  indicatorState of the VehicleControl
    //----------------------------------------
    void setIndicatorState(const VehicleControl9100::IndicatorState indicatorState)
    {
        m_delegate.setIndicatorState(indicatorState);
    }

    //========================================
    //!\brief Set the accelerationValue of the VehicleControl
    //!\param[in] accelerationValue  New accelerationValue of the VehicleControl
    //----------------------------------------
    void setAccelerationValue(const float accelerationValue) { m_delegate.setAccelerationValue(accelerationValue); }

    //========================================
    //!\brief Set doStop of the VehicleControl
    //!\param[in] doStop  of the VehicleControl
    //----------------------------------------
    void setDoStop(const bool doStop) { m_delegate.setDoStop(doStop); }

    //========================================
    //!\brief Set the stopDistance of the VehicleControl
    //!\param[in] stopDistance  New stopDistance of the VehicleControl
    //----------------------------------------
    void setStopDistance(const float stopDistance) { m_delegate.setStopDistance(stopDistance); }

public:
    //========================================
    //!\brief Get the timestamp of the VehicleControl
    //!\return timestamp of the VehicleControl
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_delegate.getTimestamp(); }

    //========================================
    //!\brief Get the sourceType of the VehicleControl
    //!\return sourceType of the VehicleControl
    //----------------------------------------
    VehicleControl9100::SourceType getSourceType() const { return m_delegate.getSourceType(); }

    //========================================
    //!\brief Get the steeringType of the VehicleControl
    //!\return steeringType of the VehicleControl
    //----------------------------------------
    VehicleControl9100::SteeringType getSteeringType() const { return m_delegate.getSteeringType(); }

    //========================================
    //!\brief Get the steeringValue of the VehicleControl
    //!\return steeringValue of the VehicleControl
    //----------------------------------------
    float getSteeringValue() const { return m_delegate.getSteeringValue(); }

    //========================================
    //!\brief Get the indicatorState of the VehicleControl
    //!\return indicatorState of the VehicleControl
    //----------------------------------------
    VehicleControl9100::IndicatorState getIndicatorState() const { return m_delegate.getIndicatorState(); }

    //========================================
    //!\brief Get the accelerationValue of the VehicleControl
    //!\return accelerationValue of the VehicleControl
    //----------------------------------------
    float getAccelerationValue() const { return m_delegate.getAccelerationValue(); }

    //========================================
    //!\brief Check if doStop is enabled
    //!\return \c True if stopped, else: \c false
    //----------------------------------------
    bool isDoStop() const { return m_delegate.isDoStop(); }

    //========================================
    //!\brief Get the stopDistance of the VehicleControl
    //!\return stopDistance of the VehicleControl
    //----------------------------------------
    float getStopDistance() const { return m_delegate.getStopDistance(); }

protected:
    VehicleControl9100 m_delegate; //< only possible specialization currently
}; // VehicleControl

//==============================================================================

bool operator==(const VehicleControl& lhs, const VehicleControl& rhs);
bool operator!=(const VehicleControl& lhs, const VehicleControl& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
//! \file
//!
//! vehicle state data type 0x2806 - send by ECU
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 14, 2013
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//------------------------------------------------------------------------------
//! \brief FUSION SYSTEM/ECU vehicle state:
//! Basic Vehicle State Data including static vehicle information (generic)
//!
//! Vehicle state data available from FUSION SYSTEM and AppBase2 (ECU).
//!
//! All angles, position and distances are given in the ISO 8855 / DIN 70000 coordinate system.
//!
//! General data type: \ref microvision::common::sdk::VehicleState
//==============================================================================
class VehicleState2806 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclestate2806"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    VehicleState2806();
    ~VehicleState2806() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    uint32_t getMicrosecondsSinceStartup() const { return m_microsecondsSinceStartup; }
    NtpTime getTimestamp() const { return m_timestamp; }
    int32_t getXPosition() const { return m_xPos; }
    int32_t getYPosition() const { return m_yPos; }
    float getCourseAngle() const { return m_courseAngle; }
    float getLongitudinalVelocity() const { return m_longitudinalVelocity; }
    float getYawRate() const { return m_yawRate; }
    float getSteeringWheelAngle() const { return m_steeringWheelAngle; }
    float getFrontWheelAngle() const { return m_frontWheelAngle; }
    float getVehicleWidth() const { return m_vehicleWidth; }
    float getVehicleFrontToFrontAxle() const { return m_vehicleFrontToFrontAxle; }
    float getFrontAxleToRearAxle() const { return m_frontAxleToRearAxle; }
    float getRearAxleToVehicleRear() const { return m_rearAxleToVehicleRear; }
    float getSteerRatioCoeff0() const { return m_steerRatioCoeff0; }
    float getSteerRatioCoeff1() const { return m_steerRatioCoeff1; }
    float getSteerRatioCoeff2() const { return m_steerRatioCoeff2; }
    float getSteerRatioCoeff3() const { return m_steerRatioCoeff3; }
    float getCrossAcceleration() const { return this->m_crossAcceleration; }
    uint16_t getBlindPredictionAge() const { return this->m_blindPredictionAge; }
    float getMinTurningCircle() const { return this->m_minTurningCircle; }
    float getRearAxleToOrigin() const { return this->m_rearAxleToOrigin; }

public: // setter
    void setMicrosecondsSinceStartup(const uint32_t microseconds) { m_microsecondsSinceStartup = microseconds; }
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }
    void setXPosition(const int32_t xPos) { m_xPos = xPos; }
    void setYPosition(const int32_t yPos) { m_yPos = yPos; }
    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }
    void setLongitudinalVelocity(const float longitudinalVelocity) { m_longitudinalVelocity = longitudinalVelocity; }
    void setYawRate(const float yawRate) { m_yawRate = yawRate; }
    void setSteeringWheelAngle(const float steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }
    void setFrontWheelAngle(const float frontWheelAngle) { m_frontWheelAngle = frontWheelAngle; }
    void setVehicleWidth(const float vehicleWidth) { m_vehicleWidth = vehicleWidth; }
    void setVehicleFrontToFrontAxle(const float vehicleFrontToFrontAxle)
    {
        m_vehicleFrontToFrontAxle = vehicleFrontToFrontAxle;
    }
    void setFrontAxleToRearAxle(const float frontAxleToRearAxle) { m_frontAxleToRearAxle = frontAxleToRearAxle; }
    void setRearAxleToVehicleRear(const float rearAxleToVehicleRear)
    {
        m_rearAxleToVehicleRear = rearAxleToVehicleRear;
    }
    void setSteerRatioCoeff0(const float steerRatioCoeff0) { m_steerRatioCoeff0 = steerRatioCoeff0; }
    void setSteerRatioCoeff1(const float steerRatioCoeff1) { m_steerRatioCoeff1 = steerRatioCoeff1; }
    void setSteerRatioCoeff2(const float steerRatioCoeff2) { m_steerRatioCoeff2 = steerRatioCoeff2; }
    void setSteerRatioCoeff3(const float steerRatioCoeff3) { m_steerRatioCoeff3 = steerRatioCoeff3; }
    void setCrossAcceleration(const float crossAcceleration) { m_crossAcceleration = crossAcceleration; }
    void setBlindPredictionAge(const uint16_t blindPredictionAge) { m_blindPredictionAge = blindPredictionAge; }
    void setMinTurningCircle(const float minTurningCircle) { m_minTurningCircle = minTurningCircle; }
    void setRearAxleToOrigin(const float rearAxleToOrigin) { m_rearAxleToOrigin = rearAxleToOrigin; }

protected:
    uint32_t m_microsecondsSinceStartup{0}; //!< microseconds since startup.
    NtpTime m_timestamp{}; //!< timestamp of this data
    int32_t m_xPos{0}; //!< Absolute X Position from origin [1/10 mm]
    int32_t m_yPos{0}; //!< Absolute Y Position from origin [1/10 mm]
    float m_courseAngle{NaN}; //!< Orientation [rad]
    float m_longitudinalVelocity{NaN}; //!< Current longitudinal velocity of the vehicle [0.01 m/s]
    float m_yawRate{NaN}; //!< Difference in Heading during Timediff [rad/s]
    float m_steeringWheelAngle{NaN}; //!< steering wheel angle [rad]
    float m_crossAcceleration{NaN}; //!< crossAcceleration of vehicle [m/s²] // (was reserved0)
    float m_frontWheelAngle{NaN}; //!< front wheel angle [rad]
    uint16_t m_blindPredictionAge{0};
    //!< blindPredictionAge // If nonzero, this VehicleStateBasic was based on prediction only instead of
    //!< measurement messages. In that case, this field counts the number of time a VehicleStateBasic had
    //!< to be predicted without measurement. Increasing numbers indicate decreasing accuracy.
    float m_vehicleWidth{NaN}; //!< Vehicle width [m]
    float m_minTurningCircle{NaN}; //!< minTurningCircle // (was reserved2)
    float m_vehicleFrontToFrontAxle{NaN}; //!< Distance: vehicle's front axle to vehicle's front [m]
    float m_frontAxleToRearAxle{NaN}; //!< Distance: vehicle's rear axle to vehicle's front axle [m]
    float m_rearAxleToVehicleRear{NaN}; //!< Distance: vehicle's rear axle to vehicle's rear [m]
    float m_rearAxleToOrigin{NaN}; //!< rearAxleToOrigin // (was reserved3)
    float m_steerRatioCoeff0{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff1{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff2{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff3{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
}; // VehicleStateBasic2806

//==============================================================================

//==============================================================================

bool operator==(const VehicleState2806& lhs, const VehicleState2806& rhs);
bool operator!=(const VehicleState2806& lhs, const VehicleState2806& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

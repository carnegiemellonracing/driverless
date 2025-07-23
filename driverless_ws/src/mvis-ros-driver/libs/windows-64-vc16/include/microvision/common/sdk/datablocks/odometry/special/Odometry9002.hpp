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
//! \date Mar 18, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Motion data
//!
//! This data type contains unprocessed motion data, such as the velocity and the yaw rate of the vehicle
//! that is provided to the system via CAN.
//!
//! General data type: \ref microvision::common::sdk::Odometry
//------------------------------------------------------------------------------
class Odometry9002 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const int nbOfReserved = 4;

public:
    using ArrayOfReserved = std::array<uint32_t, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.odometry9002"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    Odometry9002();
    ~Odometry9002() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Get the steering angle.
    //----------------------------------------
    double getSteeringAngle() const { return m_steeringAngle; }

    //========================================
    //!\brief Get the steering wheel angle in [rad].
    //----------------------------------------
    double getSteeringWheelAngle() const { return m_steeringWheelAngle; }

    //========================================
    //!\brief Get the steering wheel angle velocity.
    //----------------------------------------
    double getSteeringWheelAngleVelocity() const { return m_steeringWheelAngleVelocity; }

    //========================================
    //!\brief Get the front left wheel speed in [m/s].
    //----------------------------------------
    double getWheelSpeedFL() const { return m_wheelSpeedFL; }

    //========================================
    //!\brief Get the front right wheel speed in [m/s].
    //----------------------------------------
    double getWheelSpeedFR() const { return m_wheelSpeedFR; }

    //========================================
    //!\brief Get the rear left wheel speed in [m/s].
    //----------------------------------------
    double getWheelSpeedRL() const { return m_wheelSpeedRL; }

    //========================================
    //!\brief Get the rear right wheel speed in [m/s].
    //----------------------------------------
    double getWheelSpeedRR() const { return m_wheelSpeedRR; }

    //========================================
    //!\brief Get the wheel size.
    //----------------------------------------
    double getWheelCircumference() const { return m_wheelCircumference; }

    //========================================
    //!\brief Get the vehicle velocity in [m/s].
    //----------------------------------------
    double getVehVelocity() const { return m_vehVelocity; }

    //========================================
    //!\brief Get the vehicle acceleration in [m/s^2].
    //----------------------------------------
    double getVehAcceleration() const { return m_vehAcceleration; }

    //========================================
    //!\brief Get the vehicle yaw rate in [rad/s].
    //----------------------------------------
    double getVehYawRate() const { return m_vehYawRate; }

    //========================================
    //!\brief Get the timestamp.
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_timestamp; }

    //========================================
    //!\brief Get the wheel base in [m].
    //----------------------------------------
    float getWheelBase() const { return m_wheelBase; }

    //========================================
    //!\brief Get reserved data.
    //----------------------------------------
    const ArrayOfReserved& getReserved() const { return m_reserved; }

public:
    //========================================
    //!\brief Set the steering angle.
    //----------------------------------------
    void setSteeringAngle(const double steeringAngle) { m_steeringAngle = steeringAngle; }

    //========================================
    //!\brief Set the steering wheel angle in [rad].
    //----------------------------------------
    void setSteeringWheelAngle(const double steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }

    //========================================
    //!\brief Set the steering wheel angle velocity.
    //----------------------------------------
    void setSteeringWheelAngleVelocity(double steeringWheelAngleVelocity)
    {
        this->m_steeringWheelAngleVelocity = steeringWheelAngleVelocity;
    }

    //========================================
    //!\brief Set the front left wheel speed in [m/s].
    //----------------------------------------
    void setWheelSpeedFL(const double wheelSpeedFL) { m_wheelSpeedFL = wheelSpeedFL; }

    //========================================
    //!\brief Set the front right wheel speed in [m/s].
    //----------------------------------------
    void setWheelSpeedFR(const double wheelSpeedFR) { m_wheelSpeedFR = wheelSpeedFR; }

    //========================================
    //!\brief Set the rear left wheel speed in [m/s].
    //----------------------------------------
    void setWheelSpeedRL(const double wheelSpeedRL) { m_wheelSpeedRL = wheelSpeedRL; }

    //========================================
    //!\brief Set the rear right wheel speed in [m/s].
    //----------------------------------------
    void setWheelSpeedRR(const double wheelSpeedRR) { m_wheelSpeedRR = wheelSpeedRR; }

    //========================================
    //!\brief Set the wheel size.
    //----------------------------------------
    void setWheelCircumference(const double wheelCircumference) { m_wheelCircumference = wheelCircumference; }

    //========================================
    //!\brief Set the vehicle velocity in [m/s].
    //----------------------------------------
    void setVehVelocity(const double vehVelocity) { m_vehVelocity = vehVelocity; }

    //========================================
    //!\brief Set the vehicle acceleration in [m/s^2].
    //----------------------------------------
    void setVehAcceleration(const double vehAcceleration) { m_vehAcceleration = vehAcceleration; }

    //========================================
    //!\brief Set the vehicle yaw rate in [rad/s].
    //----------------------------------------
    void setVehYawRate(const double vehYawRate) { m_vehYawRate = vehYawRate; }

    //========================================
    //!\brief Set the timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_timestamp = timestamp; }

    //========================================
    //!\brief Set the wheel base in [m].
    //----------------------------------------
    void setWheelBase(const float wheelBase) { m_wheelBase = wheelBase; }

protected:
    double m_steeringAngle{NaN_double}; //!< Angle at which the vehicle is being steered. [rad]
    double m_steeringWheelAngle{NaN_double}; //!< Angle by which the steering wheel is rotated. [rad]
    double m_steeringWheelAngleVelocity{NaN_double}; //!< Velocity of the steering wheel angle. [rad/s]
    double m_wheelSpeedFL{NaN_double}; //!< Speed of Front Left wheel. [m/s]
    double m_wheelSpeedFR{NaN_double}; //!< Speed of Front Right wheel. [m/s]
    double m_wheelSpeedRL{NaN_double}; //!< Speed of Rear Left wheel. [m/s]
    double m_wheelSpeedRR{NaN_double}; //!< Speed of Rear Right wheel. [m/s]
    double m_wheelCircumference{NaN_double}; //!< Circumference of the wheel. [m]
    double m_vehVelocity{NaN_double}; //!< Current velocity of the vehicle. [m/s]
    double m_vehAcceleration{NaN_double}; //!< Current acceleration of the vehicle. [m/s^2]
    double m_vehYawRate{NaN_double}; //!< Current yaw rate of the vehicle. [rad/s]
    Timestamp m_timestamp{}; //!< Unique timestamp associated with each measurement signal.

    float m_wheelBase{0.0F}; //!< Wheel base [m]

private:
    ArrayOfReserved m_reserved{{0U, 0U, 0U, 0U}};
}; // Odometry9002

//==============================================================================

bool operator==(const Odometry9002& od1, const Odometry9002& od2);

bool operator!=(const Odometry9002& od1, const Odometry9002& od2);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

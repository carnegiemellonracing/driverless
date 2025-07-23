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
//! \date Jan 29th, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/datablocks/odometry/special/Odometry9002.hpp>
#include <microvision/common/sdk/datablocks/odometry/special/Odometry9003.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

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
//! Special data type: \ref microvision::common::sdk::Odometry9002
//! Special data type: \ref microvision::common::sdk::Odometry9003
//------------------------------------------------------------------------------
class Odometry final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const Odometry& lhs, const Odometry& rhs);

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.odometry"};

    //========================================
    //! \brief Get the static hash value of the class id (static version).
    //!
    //! \return The hash value specifying the custom data container class.
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    Odometry();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Odometry() override = default;

public:
    //========================================
    //! \brief Get the static hash value of the class id.
    //!
    //! \return The hash value specifying the custom data container class.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    //========================================
    //! \brief Get the timestamp.
    //!
    //! \return The unique timestamp associated with this measurement signal.
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the front axis steering angle.
    //!
    //! \return The steering angle of the front axis [rad].
    //----------------------------------------
    double getFrontAxisSteeringAngle() const { return m_frontAxisSteeringAngle; }

    //========================================
    //! \brief Get the rear axis steering angle.
    //!
    //! \return The steering angle of the rear axis [rad].
    //----------------------------------------
    double getRearAxisSteeringAngle() const { return m_rearAxisSteeringAngle; }

    //========================================
    //! \brief Get the steering wheel angle.
    //!
    //! \return The angle of steering wheel [rad].
    //----------------------------------------
    double getSteeringWheelAngle() const { return m_steeringWheelAngle; }

    //========================================
    //! \brief Get the steering wheel angular velocity.
    //!
    //! \return The rotation velocity of the steering wheel [rad/s].
    //----------------------------------------
    double getSteeringWheelAngularVelocity() const { return m_steeringWheelAngularVelocity; }

    //========================================
    //! \brief Get the steering torque.
    //!
    //! \return The steering torque for the steering column [Nm].
    //----------------------------------------
    double getSteeringTorque() const { return m_steeringTorque; }

    //========================================
    //! \brief Get the wheel speed front left.
    //!
    //! \return The velocity of the front left wheel [m/s].
    //----------------------------------------
    double getWheelSpeedFrontLeft() const { return m_wheelSpeedFrontLeft; }

    //========================================
    //! \brief Get the wheel speed front right.
    //!
    //! \return The velocity of the front right wheel [m/s].
    //----------------------------------------
    double getWheelSpeedFrontRight() const { return m_wheelSpeedFrontRight; }

    //========================================
    //! \brief Get the wheel speed rear left.
    //!
    //! \return The velocity of the rear left wheel [m/s].
    //----------------------------------------
    double getWheelSpeedRearLeft() const { return m_wheelSpeedRearLeft; }

    //========================================
    //! \brief Get the wheel speed rear right.
    //!
    //! \return The velocity of the rear right wheel [m/s].
    //----------------------------------------
    double getWheelSpeedRearRight() const { return m_wheelSpeedRearRight; }

    //========================================
    //! \brief Get the wheel circumference.
    //!
    //! \return The overall circumference of the vehicle wheels [m].
    //----------------------------------------
    double getWheelCircumference() const { return m_wheelCircumference; }

    //========================================
    //! \brief Get the wheel ticks front left.
    //!
    //! \return The ticks of the front left wheel.
    //----------------------------------------
    double getWheelTicksFrontLeft() const { return m_wheelTicksFrontLeft; }

    //========================================
    //! \brief Get the wheel ticks front right.
    //!
    //! \return The ticks of the front right wheel.
    //----------------------------------------
    double getWheelTicksFrontRight() const { return m_wheelTicksFrontRight; }

    //========================================
    //! \brief Get the wheel ticks rear left.
    //!
    //! \return The ticks of the rear left wheel.
    //----------------------------------------
    double getWheelTicksRearLeft() const { return m_wheelTicksRearLeft; }

    //========================================
    //! \brief Get the wheel ticks rear right.
    //!
    //! \return The ticks of the rear right wheel.
    //----------------------------------------
    double getWheelTicksRearRight() const { return m_wheelTicksRearRight; }

    //========================================
    //! \brief Get the vehicle velocity.
    //!
    //! \return The vehicle velocity [m/s].
    //----------------------------------------
    double getVehicleVelocity() const { return m_vehicleVelocity; }

    //========================================
    //! \brief Get the roll rate.
    //!
    //! \return The roll rate of the vehicle [rad/s].
    //----------------------------------------
    double getRollRate() const { return m_rollRate; }

    //========================================
    //! \brief Get the pitch rate.
    //!
    //! \return The pitch rate of the vehicle [rad/s].
    //----------------------------------------
    double getPitchRate() const { return m_pitchRate; }

    //========================================
    //! \brief Get the yaw rate.
    //!
    //! \return The yaw rate of the vehicle [rad/s].
    //----------------------------------------
    double getYawRate() const { return m_yawRate; }

    //========================================
    //! \brief Get the longitudinal acceleration.
    //!
    //! \return The longitudinal (x-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getLongitudinalAcceleration() const { return m_longitudinalAcceleration; }

    //========================================
    //! \brief Get the cross acceleration.
    //!
    //! \return The cross (y-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getCrossAcceleration() const { return m_crossAcceleration; }

    //========================================
    //! \brief Get the vertical acceleration.
    //!
    //! \return The vertical (z-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getVerticalAcceleration() const { return m_verticalAcceleration; }

    //========================================
    //! \brief Get the wheel base.
    //!
    //! \return The wheel base [m].
    //----------------------------------------
    float getWheelBase() const { return m_wheelBase; }

public: // setter
    //========================================
    //! \brief Set the timestamp.
    //!
    //! \param[in] timestamp The new timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_timestamp = timestamp; }

    //========================================
    //! \brief Set the front axis steering angle.
    //!
    //! \param[in] frontAxisSteeringAngle  The new front axis steering angle [rad].
    //----------------------------------------
    void setFrontAxisSteeringAngle(const double frontAxisSteeringAngle)
    {
        m_frontAxisSteeringAngle = frontAxisSteeringAngle;
    }

    //========================================
    //! \brief Set the rear axis steering angle.
    //!
    //! \param[in] rearAxisSteeringAngle  The new rear axis steering angle [rad].
    //----------------------------------------
    void setRearAxisSteeringAngle(const double rearAxisSteeringAngle)
    {
        m_rearAxisSteeringAngle = rearAxisSteeringAngle;
    }

    //========================================
    //! \brief Set the steering wheel angle.
    //!
    //! \param[in] steeringWheelAngle  The new steering wheel angle [rad] [rad].
    //----------------------------------------
    void setSteeringWheelAngle(const double steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }

    //========================================
    //! \brief Set the steering wheel angular velocity.
    //!
    //! \param[in] steeringWheelAngularVelocity  The new steering wheel angle velocity [rad/s].
    //----------------------------------------
    void setSteeringWheelAngularVelocity(const double steeringWheelAngularVelocity)
    {
        m_steeringWheelAngularVelocity = steeringWheelAngularVelocity;
    }

    //========================================
    //! \brief Set the steering torque.
    //!
    //! \param[in] steeringTorque  The new steering torque [Nm].
    //----------------------------------------
    void setSteeringTorque(const double steeringTorque) { m_steeringTorque = steeringTorque; }

    //========================================
    //! \brief Set the wheel speed front left.
    //!
    //! \param[in] wheelSpeedFrontLeft  The new wheel speed front left [m/s].
    //----------------------------------------
    void setWheelSpeedFrontLeft(const double wheelSpeedFrontLeft) { m_wheelSpeedFrontLeft = wheelSpeedFrontLeft; }

    //========================================
    //! \brief Set the wheel speed front right.
    //!
    //! \param[in] wheelSpeedFrontRight  The new wheel speed front right [m/s].
    //----------------------------------------
    void setWheelSpeedFrontRight(const double wheelSpeedFrontRight) { m_wheelSpeedFrontRight = wheelSpeedFrontRight; }

    //========================================
    //! \brief Set the wheel speed rear left.
    //!
    //! \param[in] wheelSpeedRearLeft  The new wheel speed rear left [m/s].
    //----------------------------------------
    void setWheelSpeedRearLeft(const double wheelSpeedRearLeft) { m_wheelSpeedRearLeft = wheelSpeedRearLeft; }

    //========================================
    //! \brief Set the wheel speed rear right.
    //!
    //! \param[in] wheelSpeedRearRight  The new wheel speed rear right [m/s].
    //----------------------------------------
    void setWheelSpeedRearRight(const double wheelSpeedRearRight) { m_wheelSpeedRearRight = wheelSpeedRearRight; }

    //========================================
    //! \brief Set the wheel circumference.
    //!
    //! \param[in] wheelCircumference  The new wheel circumference [m].
    //----------------------------------------
    void setWheelCircumference(const double wheelCircumference) { m_wheelCircumference = wheelCircumference; }

    //========================================
    //! \brief Set the wheel ticks front left.
    //!
    //! \param[in] wheelTicksFrontLeft  The new wheel ticks front left.
    //----------------------------------------
    void setWheelTicksFrontLeft(const double wheelTicksFrontLeft) { m_wheelTicksFrontLeft = wheelTicksFrontLeft; }

    //========================================
    //! \brief Set the wheel ticks front right.
    //!
    //! \param[in] wheelTicksFrontRight  The new wheel ticks front right.
    //----------------------------------------
    void setWheelTicksFrontRight(const double wheelTicksFrontRight) { m_wheelTicksFrontRight = wheelTicksFrontRight; }

    //========================================
    //! \brief Set the wheel ticks rear left.
    //!
    //! \param[in] wheelTicksRearLeft  The new wheel ticks rear left.
    //----------------------------------------
    void setWheelTicksRearLeft(const double wheelTicksRearLeft) { m_wheelTicksRearLeft = wheelTicksRearLeft; }

    //========================================
    //! \brief Set the wheel ticks rear right.
    //!
    //! \param[in] wheelTicksRearRight  The new wheel ticks rear right.
    //----------------------------------------
    void setWheelTicksRearRight(const double wheelTicksRearRight) { m_wheelTicksRearRight = wheelTicksRearRight; }

    //========================================
    //! \brief Set the vehicle velocity.
    //!
    //! \param[in] vehicleVelocity  The new vehicle velocity [m/s].
    //----------------------------------------
    void setVehicleVelocity(const double vehicleVelocity) { m_vehicleVelocity = vehicleVelocity; }

    //========================================
    //! \brief Set the roll rate.
    //!
    //! \param[in] rollRate  The new roll rate [rad/s].
    //----------------------------------------
    void setRollRate(const double rollRate) { m_rollRate = rollRate; }

    //========================================
    //! \brief Set the pitch rate.
    //!
    //! \param[in] pitchRate  The new pitch rate [rad/s].
    //----------------------------------------
    void setPitchRate(const double pitchRate) { m_pitchRate = pitchRate; }

    //========================================
    //! \brief Set the yaw rate.
    //!
    //! \param[in] yawRate  The new yaw rate [rad/s].
    //----------------------------------------
    void setYawRate(const double yawRate) { m_yawRate = yawRate; }

    //========================================
    //! \brief Set the longitudinal acceleration.
    //!
    //! \param[in] longitudinalAcceleration  The new longitudinal acceleration [m/(s^2)].
    //----------------------------------------
    void setLongitudinalAcceleration(const double longitudinalAcceleration)
    {
        m_longitudinalAcceleration = longitudinalAcceleration;
    }

    //========================================
    //! \brief Set the cross acceleration.
    //!
    //! \param[in] crossAcceleration  The new cross acceleration [m/(s^2)].
    //----------------------------------------
    void setCrossAcceleration(const double crossAcceleration) { m_crossAcceleration = crossAcceleration; }

    //========================================
    //! \brief Set the vertical acceleration.
    //!
    //! \param[in] verticalAcceleration  The new vertical acceleration [m/(s^2)].
    //----------------------------------------
    void setVerticalAcceleration(const double verticalAcceleration) { m_verticalAcceleration = verticalAcceleration; }

    //========================================
    //! \brief Set the wheel base.
    //!
    //! \param[in] wheelBase  The new wheel base [m].
    //----------------------------------------
    void setWheelBase(const float wheelBase) { m_wheelBase = wheelBase; }

private:
    //========================================
    //! \brief Fill the fields of this general data container from a special data container.
    //!
    //! \param[in] odometry9002  Special data container to read from.
    //----------------------------------------
    void setFrom(const Odometry9002& odometry9002);

    //========================================
    //! \brief Fill the fields of a special data container from this general data container.
    //!
    //! \param[out] odometry9002  Special data container to write to.
    //----------------------------------------
    void setTo(Odometry9002& odometry9002) const;

    //========================================
    //! \brief Fill the fields of this general data container from a special data container.
    //!
    //! \param[in] odometry9003  Special data container to read from.
    //----------------------------------------
    void setFrom(const Odometry9003& odometry9003);

    //========================================
    //! \brief Fill the fields of a special data container from this general data container.
    //!
    //! \param[out] odometry9003  Special data container to write to.
    //----------------------------------------
    void setTo(Odometry9003& odometry9003) const;

protected:
    Timestamp m_timestamp{};
    double m_frontAxisSteeringAngle{NaN_double}; // [rad]
    double m_rearAxisSteeringAngle{NaN_double}; // [rad]
    double m_steeringWheelAngle{NaN_double}; // [rad]
    double m_steeringWheelAngularVelocity{NaN_double}; // [rad/s]
    double m_steeringTorque{NaN_double}; //[Nm]
    double m_wheelSpeedFrontLeft{NaN_double}; //[m/s]
    double m_wheelSpeedFrontRight{NaN_double}; // [m/s]
    double m_wheelSpeedRearLeft{NaN_double}; // [m/s]
    double m_wheelSpeedRearRight{NaN_double}; // [m/s]
    double m_wheelCircumference{NaN_double}; // [m]
    double m_wheelTicksFrontLeft{NaN_double};
    double m_wheelTicksFrontRight{NaN_double};
    double m_wheelTicksRearLeft{NaN_double};
    double m_wheelTicksRearRight{NaN_double};
    double m_vehicleVelocity{NaN_double}; // [m/s]
    double m_rollRate{NaN_double}; // [rad/s]
    double m_pitchRate{NaN_double}; // [rad/s]
    double m_yawRate{NaN_double}; // [rad/s]
    double m_longitudinalAcceleration{NaN_double}; // [m/(s^2)]
    double m_crossAcceleration{NaN_double}; // [m/(s^2)]
    double m_verticalAcceleration{NaN_double}; // [m/(s^2)]
    float m_wheelBase{0.0F}; // [m]
}; // Odometry

//==============================================================================

//==============================================================================
//! \brief Test odometry objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const Odometry& lhs, const Odometry& rhs);

//==============================================================================
//! \brief Test odometry objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Odometry& lhs, const Odometry& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

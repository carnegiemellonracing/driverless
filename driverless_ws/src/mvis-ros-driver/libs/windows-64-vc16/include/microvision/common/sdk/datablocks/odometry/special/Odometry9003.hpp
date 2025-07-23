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
//! \date Nov 21, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/odometry/special/OdometryParameterIn9003.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

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
//! In contrast to the data type Odometry9002 that has a dedicated field for every parameter and thus transmits
//! the whole set for every change, the Odometry9003 is made of a list that contains only those parameters that
//! have changed since last time the data type was sent. So, the Odometry9003 data type has a smaller memory
//! footprint than the Odometry9002.
//!
//! General data type: \ref microvision::common::sdk::Odometry
//------------------------------------------------------------------------------
class Odometry9003 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using OdometryParameterMap = std::map<OdometryParameterIn9003::ParameterKey, OdometryParameterIn9003>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.odometry9003"};

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
    Odometry9003();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~Odometry9003() override = default;

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
    Timestamp getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the number of parameters.
    //!
    //! \return The number of parameters in the map.
    //----------------------------------------
    uint32_t getNbOfParameters() const { return static_cast<uint32_t>(m_parameterMap.size()); }

    //========================================
    //! \brief Get the parameter map.
    //!
    //! \return The map with all parameters of this measurement.
    //----------------------------------------
    const OdometryParameterMap& getParameterMap() const { return m_parameterMap; }

    //========================================
    //! \brief Get the parameter map.
    //!
    //! \return The map with all parameters of this measurement.
    //----------------------------------------
    OdometryParameterMap& getParameterMap() { return m_parameterMap; }

public: // parameter getter (high level)
    //========================================
    //! \brief Get the steering wheel angle.
    //!
    //! \return The angle of steering wheel [rad].
    //----------------------------------------
    double getSteeringWheelAngle() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringWheelAngle, value) ? value
                                                                                                      : NaN_double;
    }

    //========================================
    //! \brief Get the steering wheel angular velocity.
    //!
    //! \return The rotation velocity of the steering wheel [rad/s].
    //----------------------------------------
    double getSteeringWheelAngularVelocity() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringWheelAngularVelocity, value)
                   ? value
                   : NaN_double;
    }

    //========================================
    //! \brief Get the front axis steering angle.
    //!
    //! \return The steering angle of the front axis [rad].
    //----------------------------------------
    double getFrontAxisSteeringAngle() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::FrontAxisSteeringAngle, value) ? value
                                                                                                          : NaN_double;
    }

    //========================================
    //! \brief Get the rear axis steering angle.
    //!
    //! \return The steering angle of the rear axis [rad].
    //----------------------------------------
    double getRearAxisSteeringAngle() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::RearAxisSteeringAngle, value) ? value
                                                                                                         : NaN_double;
    }

    //========================================
    //! \brief Get the steering torque.
    //!
    //! \return The steering torque for the steering column [Nm].
    //----------------------------------------
    double getSteeringTorque() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringTorque, value) ? value : NaN_double;
    }

    //========================================
    //! \brief Get the wheel speed front left.
    //!
    //! \return The velocity of the front left wheel [m/s].
    //----------------------------------------
    double getWheelSpeedFrontLeft() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontLeft, value) ? value
                                                                                                       : NaN_double;
    }

    //========================================
    //! \brief Get the wheel speed front right.
    //!
    //! \return The velocity of the front right wheel [m/s].
    //----------------------------------------
    double getWheelSpeedFrontRight() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontRight, value) ? value
                                                                                                        : NaN_double;
    }

    //========================================
    //! \brief Get the wheel speed rear left.
    //!
    //! \return The velocity of the rear left wheel [m/s].
    //----------------------------------------
    double getWheelSpeedRearLeft() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedRearLeft, value) ? value
                                                                                                      : NaN_double;
    }

    //========================================
    //! \brief Get the wheel speed rear right.
    //!
    //! \return The velocity of the rear right wheel [m/s].
    //----------------------------------------
    double getWheelSpeedRearRight() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedRearRight, value) ? value
                                                                                                       : NaN_double;
    }

    //========================================
    //! \brief Get the wheel circumference.
    //!
    //! \return The overall circumference of the vehicle wheels [m].
    //----------------------------------------
    double getWheelCircumference() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelCircumference, value) ? value
                                                                                                      : NaN_double;
    }

    //========================================
    //! \brief Get the wheel ticks front left.
    //!
    //! \return The ticks of the front left wheel.
    //----------------------------------------
    double getWheelTicksFrontLeft() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksFrontLeft, value) ? value
                                                                                                       : NaN_double;
    }

    //========================================
    //! \brief Get the wheel ticks front right.
    //!
    //! \return The ticks of the front right wheel.
    //----------------------------------------
    double getWheelTicksFrontRight() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksFrontRight, value) ? value
                                                                                                        : NaN_double;
    }

    //========================================
    //! \brief Get the wheel ticks rear left.
    //!
    //! \return The ticks of the rear left wheel.
    //----------------------------------------
    double getWheelTicksRearLeft() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksRearLeft, value) ? value
                                                                                                      : NaN_double;
    }

    //========================================
    //! \brief Get the wheel ticks rear right.
    //!
    //! \return The ticks of the rear right wheel.
    //----------------------------------------
    double getWheelTicksRearRight() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksRearRight, value) ? value
                                                                                                       : NaN_double;
    }

    //========================================
    //! \brief Get the vehicle velocity.
    //!
    //! \return The vehicle velocity [m/s].
    //----------------------------------------
    double getVehicleVelocity() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::VehicleVelocity, value) ? value : NaN_double;
    }

    //========================================
    //! \brief Get the roll rate.
    //!
    //! \return The roll rate of the vehicle [rad/s].
    //----------------------------------------
    double getRollRate() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::RollRate, value) ? value : NaN_double;
    }

    //========================================
    //! \brief Get the pitch rate.
    //!
    //! \return The pitch rate of the vehicle [rad/s].
    //----------------------------------------
    double getPitchRate() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::PitchRate, value) ? value : NaN_double;
    }

    //========================================
    //! \brief Get the yaw rate.
    //!
    //! \return The yaw rate of the vehicle [rad/s].
    //----------------------------------------
    double getYawRate() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::YawRate, value) ? value : NaN_double;
    }

    //========================================
    //! \brief Get the longitudinal acceleration.
    //!
    //! \return The longitudinal (x-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getLongitudinalAcceleration() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::LongitudinalAcceleration, value)
                   ? value
                   : NaN_double;
    }

    //========================================
    //! \brief Get the cross acceleration.
    //!
    //! \return The cross (y-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getCrossAcceleration() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::CrossAcceleration, value) ? value
                                                                                                     : NaN_double;
    }

    //========================================
    //! \brief Get the vertical acceleration.
    //!
    //! \return The vertical (z-direction) acceleration of the vehicle [m/(s^2)].
    //----------------------------------------
    double getVerticalAcceleration() const
    {
        double value{NaN_double};
        return getParameter<double>(OdometryParameterIn9003::ParameterKey::VerticalAcceleration, value) ? value
                                                                                                        : NaN_double;
    }

public: // parameter getter (low level)
    //========================================
    //! \brief Get the value of a parameter from the map.
    //!
    //! \tparam T                   Type of parameter value to get.
    //! \param[in]  parameterKey    Key of parameter to get.
    //! \param[out] parameterValue  Value as retrieved from the map.
    //! \return \c True, if the parameter is in the map and has the desired type, \c false otherwise.
    //----------------------------------------
    template<typename T>
    bool getParameter(const OdometryParameterIn9003::ParameterKey parameterKey, T& parameterValue) const
    {
        const auto iter = m_parameterMap.find(parameterKey);
        if ((iter == m_parameterMap.end()) || (iter->second.isOfType<T>() == false))
        {
            // Not found or wrong type.
            return false;
        }
        else
        {
            parameterValue = iter->second.getValue<T>();
            return true;
        }
    }

    //========================================
    //! \brief Get a parameter from the map.
    //!
    //! \param[in]  parameterKey  Type of parameter to get.
    //! \param[out] parameter     Parameter as retrieved from the map.
    //! \return \c True, if the parameter is in the map, \c false otherwise.
    //----------------------------------------
    bool getParameter(const OdometryParameterIn9003::ParameterKey parameterKey,
                      OdometryParameterIn9003& parameter) const
    {
        const auto iter = m_parameterMap.find(parameterKey);
        if (iter == m_parameterMap.end())
        {
            return false;
        }
        else
        {
            parameter = iter->second;
            return true;
        }
    }

public: // setter
    //========================================
    //! \brief Set the timestamp.
    //!
    //! \param[in] timestamp The new timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_timestamp = timestamp; }

    //========================================
    //! \brief Set the parameter map.
    //!
    //! \param[in] parameterMap  The new parameter map.
    //----------------------------------------
    void setParameterMap(const OdometryParameterMap& parameterMap) { m_parameterMap = parameterMap; }

public: // parameter setter (high level)
    //========================================
    //! \brief Set the steering wheel angle.
    //!
    //! \param[in] steeringWheelAngle  The new steering wheel angle [rad].
    //!
    //! \note If the steeringWheelAngle is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setSteeringWheelAngle(const double steeringWheelAngle)
    {
        if (std::isnan(steeringWheelAngle))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::SteeringWheelAngle);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringWheelAngle, steeringWheelAngle);
        }
    }

    //========================================
    //! \brief Set the steering wheel angular velocity.
    //!
    //! \param[in] steeringWheelAngularVelocity  The new steering wheel angular velocity [rad/s].
    //!
    //! \note If the steeringWheelAngularVelocity is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setSteeringWheelAngularVelocity(const double steeringWheelAngularVelocity)
    {
        if (std::isnan(steeringWheelAngularVelocity))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::SteeringWheelAngularVelocity);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringWheelAngularVelocity,
                                 steeringWheelAngularVelocity);
        }
    }

    //========================================
    //! \brief Set the front axis steering angle.
    //!
    //! \param[in] frontAxisSteeringAngle  The new front axis steering angle [rad].
    //!
    //! \note If the frontAxisSteeringAngle is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setFrontAxisSteeringAngle(const double frontAxisSteeringAngle)
    {
        if (std::isnan(frontAxisSteeringAngle))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::FrontAxisSteeringAngle);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::FrontAxisSteeringAngle, frontAxisSteeringAngle);
        }
    }

    //========================================
    //! \brief Set the rear axis steering angle.
    //!
    //! \param[in] rearAxisSteeringAngle  The new rear axis steering angle [rad].
    //!
    //! \note If the rearAxisSteeringAngle is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setRearAxisSteeringAngle(const double rearAxisSteeringAngle)
    {
        if (std::isnan(rearAxisSteeringAngle))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::RearAxisSteeringAngle);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::RearAxisSteeringAngle, rearAxisSteeringAngle);
        }
    }

    //========================================
    //! \brief Set the steering torque.
    //!
    //! \param[in] steeringTorque  The new steering torque [Nm].
    //!
    //! \note If the steeringTorque is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setSteeringTorque(const double steeringTorque)
    {
        if (std::isnan(steeringTorque))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::SteeringTorque);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::SteeringTorque, steeringTorque);
        }
    }

    //========================================
    //! \brief Set the wheel speed front left.
    //!
    //! \param[in] wheelSpeedFrontLeft  The new wheel speed front left [m/s].
    //!
    //! \note If the wheelSpeedFrontLeft is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelSpeedFrontLeft(const double wheelSpeedFrontLeft)
    {
        if (std::isnan(wheelSpeedFrontLeft))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontLeft);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontLeft, wheelSpeedFrontLeft);
        }
    }

    //========================================
    //! \brief Set the wheel speed front right.
    //!
    //! \param[in] wheelSpeedFrontRight  The new wheel speed front right [m/s].
    //!
    //! \note If the wheelSpeedFrontRight is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelSpeedFrontRight(const double wheelSpeedFrontRight)
    {
        if (std::isnan(wheelSpeedFrontRight))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontRight);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedFrontRight, wheelSpeedFrontRight);
        }
    }

    //========================================
    //! \brief Set the wheel speed rear left.
    //!
    //! \param[in] wheelSpeedRearLeft  The new wheel speed rear left [m/s].
    //!
    //! \note If the wheelSpeedRearLeft is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelSpeedRearLeft(const double wheelSpeedRearLeft)
    {
        if (std::isnan(wheelSpeedRearLeft))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelSpeedRearLeft);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedRearLeft, wheelSpeedRearLeft);
        }
    }

    //========================================
    //! \brief Set the wheel speed rear right.
    //!
    //! \param[in] wheelSpeedRearRight  The new wheel speed rear right [m/s].
    //!
    //! \note If the wheelSpeedRearRight is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelSpeedRearRight(const double setWheelSpeedRearRight)
    {
        if (std::isnan(setWheelSpeedRearRight))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelSpeedRearRight);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelSpeedRearRight, setWheelSpeedRearRight);
        }
    }

    //========================================
    //! \brief Set the wheel circumference.
    //!
    //! \param[in] wheelCircumference  The new wheel circumference [m].
    //!
    //! \note If the wheelCircumference is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelCircumference(const double wheelCircumference)
    {
        if (std::isnan(wheelCircumference))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelCircumference);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelCircumference, wheelCircumference);
        }
    }

    //========================================
    //! \brief Set the wheel ticks front left.
    //!
    //! \param[in] wheelTicksFrontLeft  The new wheel ticks front left.
    //!
    //! \note If the wheelTicksFrontLeft is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelTicksFrontLeft(const double wheelTicksFrontLeft)
    {
        if (std::isnan(wheelTicksFrontLeft))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelTicksFrontLeft);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksFrontLeft, wheelTicksFrontLeft);
        }
    }

    //========================================
    //! \brief Set the wheel ticks front right.
    //!
    //! \param[in] wheelTicksFrontRight  The new wheel ticks front right.
    //!
    //! \note If the wheelTicksFrontRight is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelTicksFrontRight(const double wheelTicksFrontRight)
    {
        if (std::isnan(wheelTicksFrontRight))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelTicksFrontRight);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksFrontRight, wheelTicksFrontRight);
        }
    }

    //========================================
    //! \brief Set the wheel ticks rear left.
    //!
    //! \param[in] wheelTicksRearLeft  The new wheel ticks rear left.
    //!
    //! \note If the wheelTicksRearLeft is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelTicksRearLeft(const double wheelTicksRearLeft)
    {
        if (std::isnan(wheelTicksRearLeft))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelTicksRearLeft);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksRearLeft, wheelTicksRearLeft);
        }
    }

    //========================================
    //! \brief Set the wheel ticks rear right.
    //!
    //! \param[in] wheelTicksRearRight  The new wheel ticks rear right.
    //!
    //! \note If the wheelTicksRearRight is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setWheelTicksRearRight(const double wheelTicksRearRight)
    {
        if (std::isnan(wheelTicksRearRight))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::WheelTicksRearRight);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::WheelTicksRearRight, wheelTicksRearRight);
        }
    }

    //========================================
    //! \brief Set the vehicle velocity.
    //!
    //! \param[in] vehicleVelocity  The new vehicle velocity [m/s].
    //!
    //! \note If the vehicleVelocity is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setVehicleVelocity(const double vehicleVelocity)
    {
        if (std::isnan(vehicleVelocity))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::VehicleVelocity);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::VehicleVelocity, vehicleVelocity);
        }
    }

    //========================================
    //! \brief Set the roll rate.
    //!
    //! \param[in] rollRate  The new roll rate [rad/s].
    //!
    //! \note If the rollRate is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setRollRate(const double rollRate)
    {
        if (std::isnan(rollRate))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::RollRate);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::RollRate, rollRate);
        }
    }

    //========================================
    //! \brief Set the pitch rate.
    //!
    //! \param[in] pitchRate  The new pitch rate [rad/s].
    //!
    //! \note If the pitchRate is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setPitchRate(const double pitchRate)
    {
        if (std::isnan(pitchRate))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::PitchRate);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::PitchRate, pitchRate);
        }
    }

    //========================================
    //! \brief Set the yaw rate.
    //!
    //! \param[in] yawRate  The new yaw rate [rad/s].
    //!
    //! \note If the yawRate is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setYawRate(const double yawRate)
    {
        if (std::isnan(yawRate))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::YawRate);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::YawRate, yawRate);
        }
    }

    //========================================
    //! \brief Set the longitudinal acceleration.
    //!
    //! \param[in] longitudinalAcceleration  The new longitudinal acceleration [m/(s^2)].
    //!
    //! \note If the longitudinalAcceleration is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setLongitudinalAcceleration(const double longitudinalAcceleration)
    {
        if (std::isnan(longitudinalAcceleration))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::LongitudinalAcceleration);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::LongitudinalAcceleration,
                                 longitudinalAcceleration);
        }
    }

    //========================================
    //! \brief Set the cross acceleration.
    //!
    //! \param[in] crossAcceleration  The new cross acceleration [m/(s^2)].
    //!
    //! \note If the crossAcceleration is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setCrossAcceleration(const double crossAcceleration)
    {
        if (std::isnan(crossAcceleration))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::CrossAcceleration);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::CrossAcceleration, crossAcceleration);
        }
    }

    //========================================
    //! \brief Set the vertical acceleration.
    //!
    //! \param[in] verticalAcceleration  The new vertical acceleration [m/(s^2)].
    //!
    //! \note If the verticalAcceleration is set to \c NaN the parameter is removed from the map.
    //----------------------------------------
    void setVerticalAcceleration(const double verticalAcceleration)
    {
        if (std::isnan(verticalAcceleration))
        {
            removeParameter(OdometryParameterIn9003::ParameterKey::VerticalAcceleration);
        }
        else
        {
            setParameter<double>(OdometryParameterIn9003::ParameterKey::VerticalAcceleration, verticalAcceleration);
        }
    }

public: // parameter setter (low level)
    //========================================
    //! \brief Set the value of a parameter in the map.
    //!
    //! \tparam T                  Type of parameter value to set.
    //! \param[in] parameterKey    Key of parameter to set.
    //! \param[in] parameterValue  Parameter value to set.
    //----------------------------------------
    template<typename T>
    void setParameter(const OdometryParameterIn9003::ParameterKey parameterKey, const T& parameterValue)
    {
        setParameter(OdometryParameterIn9003(parameterKey, parameterValue));
    }

    //========================================
    //! \brief Set a parameter in the map.
    //!
    //! \param[in] parameter  Parameter to set.
    //----------------------------------------
    void setParameter(const OdometryParameterIn9003& parameter) { m_parameterMap[parameter.getKey()] = parameter; }

    //========================================
    //! \brief Remove a parameter from the map.
    //!
    //! \param[in] parameterType  Type of parameter to remove.
    //----------------------------------------
    void removeParameter(const OdometryParameterIn9003::ParameterKey parameterType)
    {
        m_parameterMap.erase(parameterType);
    }

    //========================================
    //! \brief Remove a parameter from the map.
    //!
    //! \param[in] parameter  Parameter to remove.
    //----------------------------------------
    void removeParameter(const OdometryParameterIn9003& parameter) { removeParameter(parameter.getKey()); }

protected:
    Timestamp m_timestamp{};
    OdometryParameterMap m_parameterMap;
}; // Odometry9003

//==============================================================================

//==============================================================================
//! \brief Test odometry 9003 objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const Odometry9003& lhs, const Odometry9003& rhs);

//==============================================================================
//! \brief Test odometry 9003 objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const Odometry9003& lhs, const Odometry9003& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

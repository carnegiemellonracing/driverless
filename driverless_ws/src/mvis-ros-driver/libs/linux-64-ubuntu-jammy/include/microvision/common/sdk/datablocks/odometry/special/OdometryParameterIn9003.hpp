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

#include <microvision/common/logging/logging.hpp>
#include <microvision/common/sdk/misc/Any.hpp>

#include <iostream>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Possible parameters in an Odometry9003 container.
//------------------------------------------------------------------------------
class OdometryParameterIn9003 final
{
public:
    //========================================
    //! \brief Possible parameter keys.
    //----------------------------------------
    enum class ParameterKey : uint16_t
    {
        SteeringWheelAngle           = 0, //!< angle of steering wheel [rad]
        SteeringWheelAngularVelocity = 1, //!< rotation velocity of the steering wheel [rad/s]
        FrontAxisSteeringAngle       = 10, //!< steering angle of the front axis [rad]
        RearAxisSteeringAngle        = 11, //!< steering angle of the rear axis [rad]
        SteeringTorque               = 12, //!< The steering torque for the steering column [Nm]
        WheelSpeedFrontLeft          = 20, //!< velocity of the front left wheel [m/s]
        WheelSpeedFrontRight         = 21, //!< velocity of the front right wheel [m/s]
        WheelSpeedRearLeft           = 22, //!< velocity of the rear left wheel [m/s]
        WheelSpeedRearRight          = 23, //!< velocity of the rear right wheel [m/s]
        WheelCircumference           = 24, //!< overall circumference of the vehicle wheels [m]
        WheelTicksFrontLeft          = 25, //!< wheel ticks of the front left wheel
        WheelTicksFrontRight         = 26, //!< wheel ticks of the front right wheel
        WheelTicksRearLeft           = 27, //!< wheel ticks of the rear left wheel
        WheelTicksRearRight          = 28, //!< wheel ticks od the rear right wheel
        VehicleVelocity              = 30, //!< vehicle velocity [m/s]
        RollRate                     = 40, //!< roll rate of the vehicle [rad/s]
        PitchRate                    = 41, //!< pitch rate of the vehicle [rad/s]
        YawRate                      = 42, //!< yaw rate of the vehicle [rad/s]
        LongitudinalAcceleration     = 53, //!< longitudinal (x-direction) acceleration of the vehicle [m/(s*s)]
        CrossAcceleration            = 54, //!< cross (y-direction) acceleration of the vehicle [m/(s*s)]
        VerticalAcceleration         = 55, //!< vertical (z-direction) acceleration of the vehicle [m/(s*s)]

        Undefined = std::numeric_limits<uint16_t>::max() //!< Type not known/set yet.
    };

    //========================================
    //! \brief Possible parameter value types.
    //----------------------------------------
    enum class ParameterValueType : uint16_t
    {
        Undefined = 0,
        Float     = 1,
        Double    = 2,
        Int8      = 3,
        UInt8     = 4,
        Int16     = 5,
        UInt16    = 6,
        Int32     = 7,
        UInt32    = 8,
        Int64     = 9,
        UInt64    = 10,
        Bool      = 11
    };

public:
    //========================================
    //! \brief Constructor.
    //!
    //! Initializes the parameter with default (invalid) values.
    //----------------------------------------
    OdometryParameterIn9003() = default;

    //========================================
    //! \brief Constructor.
    //!
    //! Initializes the parameter with the given type and value.
    //!
    //! \tparam T         Type of parameter value.
    //! \param[in] key    Parameter key.
    //! \param[in] value  Parameter value.
    //----------------------------------------
    template<typename T>
    OdometryParameterIn9003(const ParameterKey key, const T& value)
    {
        setKey(key);
        setValue(value);
    }

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    virtual ~OdometryParameterIn9003() = default;

public:
    //========================================
    //! \brief Get the size in bytes that this meta information occupies when being serialized.
    //!
    //! \return The number of bytes used for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize() const;

    //========================================
    //! \brief Read data from the given stream and fill this meta information (deserialization).
    //!
    //! \param[in, out] is      Input data stream
    //! \return \c True if deserialization succeeds, \c false otherwise.
    //----------------------------------------
    virtual bool deserialize(std::istream& is);

    //========================================
    //! \brief Convert this meta information to a serializable format and write it to the given stream (serialization).
    //!
    //! \param[in, out] os      Output data stream
    //! \return \c True if serialization succeeds, \c false otherwise.
    //----------------------------------------
    virtual bool serialize(std::ostream& os) const;

public: // getter
    ParameterKey getKey() const { return m_key; }

    //========================================
    //! \brief Get the value of this property.
    //!
    //! \tparam T Type of the property value.
    //! \return The value of this property.
    //!
    //! \note If the value cannot be casted to the given type a \c BadAnyCast exception is thrown.
    //----------------------------------------
    template<typename T>
    T getValue() const
    {
        return anyCast<T>(m_data);
    }

    //========================================
    //! \brief Get the value of this property.
    //!
    //! \tparam    T             Type of the property value.
    //! \param[in] defaultValue  The value that is used if the property cannot be casted to the given type.
    //! \return The value of this property or the given default value if the property cannot be casted to the given type.
    //----------------------------------------
    template<typename T>
    T getValue(const T& defaultValue) const
    {
        T result;
        try
        {
            result = anyCast<T>(m_data);
        }
        catch (const BadAnyCast&)
        {
            result = defaultValue;
        }

        return result;
    }

    //========================================
    //! \brief Check if the property value is of the given type.
    //!
    //! \tparam T  Type of the property value to check.
    //! \return \c True, if the property value is of the given type, \c false otherwise.
    //----------------------------------------
    template<typename T>
    bool isOfType() const
    {
        return m_data.type() == typeid(T);
    }

public: // setter
    //========================================
    //! \brief Set the key of this property.
    //!
    //! \param[in] value  The new key of this property.
    //----------------------------------------
    void setKey(const ParameterKey key) { m_key = key; }

    //========================================
    //! \brief Set the value of this property.
    //!
    //! \tparam    T      Type of the property value.
    //! \param[in] value  The new value of this property.
    //----------------------------------------
    template<typename T>
    void setValue(const T& data)
    {
        m_data = data;
    }

    //========================================
    //! \brief Clear the value of this property.
    //----------------------------------------
    void resetValue() { m_data = Any(); }

public:
    //========================================
    //! \brief Tests this parameter for equality.
    //!
    //! \param[in] other  The other parameter to compare with.
    //! \return \c True, if the two parameters are equal, \c false otherwise.
    //----------------------------------------
    bool isEqual(const OdometryParameterIn9003& other) const;

private:
    //========================================
    //! \brief Convert a type info to a parameter value type.
    //!
    //! \param[in] typeInfo  Type info to convert.
    //! \return The corresponding parameter value type or \a ParameterValueType::Undefined if the type info is not
    //!         valid.
    //----------------------------------------
    static ParameterValueType typeInfoToType(const std::type_info& typeInfo);

    //========================================
    //! \brief Get the size of the parameter value for serialization.
    //!
    //! \return The size of the parameter value for serialization.
    //----------------------------------------
    std::size_t getDataSize() const;

private:
    ParameterKey m_key{ParameterKey::Undefined}; //!< Key of odometry parameter.
    Any m_data; //!< Value of the odometry parameter.

    static constexpr const char* loggerId = "microvision::common::sdk::OdometryParameterIn9003";
    static microvision::common::logging::LoggerSPtr logger;
}; // OdometryParameterIn9003

//==============================================================================

//==============================================================================
//! \brief Test odometry parameter objects for equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator==(const OdometryParameterIn9003& lhs, const OdometryParameterIn9003& rhs)
{
    return lhs.isEqual(rhs);
}

//==============================================================================
//! \brief Test odometry parameter objects for in-equality.
//!
//! \param[in] lhs  left object to compare
//! \param[in] rhs  right object to compare
//! \return \c True if the objects are not equal, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const OdometryParameterIn9003& lhs, const OdometryParameterIn9003& rhs)
{
    return !lhs.isEqual(rhs);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

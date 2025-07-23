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
//! \date Jul 29, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/Unconvertable.hpp>

#include <microvision/common/sdk/io.hpp>
#include <microvision/common/sdk/bufferIO.hpp>

#include <boost/functional/hash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class MeasurementIn2821;

//==============================================================================

class MeasurementKeyIn2821 final : public ComparableUnconvertable<uint16_t>
{
public:
    static bool compare(const MeasurementIn2821& m, const MeasurementKeyIn2821 key);

public:
    //========================================
    //! \brief Ids of some DataTypes. For convenience.
    //----------------------------------------
    enum class Value : uint16_t
    {
        Undefined = 0,

        VelocityX                = 1, //!< [m/s] in vehicle coordinate system ("Forward velocity"); (double)
        YawRate                  = 2, //!< [rad/s] in vehicle coordinate system; (double)
        CrossAcceleration        = 4, //!< [m/s^2] in vehicle coordinate system; (double)
        LongitudinalAcceleration = 6, //!< [m/s^2] in vehicle coordinate system; (double)
        RollAngle                = 7, //!< [rad]; (double)
        PitchAngle               = 8, //!< [rad]; (double)

        //32 - 42 reserved

        VerticalAcceleration = 43, //!< [m/s^2] in vehicle coordinate system; (double)
        PitchRate            = 44, //!< [rad/s] in vehicle coordinate system; (double)
        RollRate             = 45, //!< [rad/s] in vehicle coordinate system; (double)
        //47 reserved
        //48 reserved
        VelocityNorth = 49, //!< [m/s]; (double)
        VelocityUp    = 50, //!< [m/s]; (double)
        VelocityWest  = 51, //!< [m/s]; (double)

        Latitude  = 60, //!< [rad] WGS84-Latitude; (double)
        Longitude = 61, //!< [rad] WGS84-Longitude; (double)
        Altitude  = 62, //!< [m] Height above sea level; (double)
        //63-66 reserved
        YawAngle = 67, //!< [rad]; (double)
        //68 reserved
        //69 reserved
        UtcHours        = 70, //!< (uint8_t)
        UtcMinutes      = 71, //!< (uint8_t)
        UtcSeconds      = 72, //!< (uint8_t)
        UtcMilliSeconds = 73, //!< (uint32_t)
        //74 reserved
        //76 reserved
        //77 reserved
        //78 reserved
        UtcDays   = 79, //!< (uint8_t)
        UtcMonths = 80, //!< (uint8_t)
        UtcYears  = 81, //!< (uint16_t)
        //82-86 reserved
        //100 reserved
        //200-209 reserved
        //300 reserved
        //301 reserved

        // Values from 400 to 4xx are specific to ECU Extended Tracking.
        // Corresponding declarations can be found in ObjectIn2281.hpp.

        //400-407 reserved
        //501-504 reserved
        //510-549 reserved

        // (Object) Label
        LabelUserData = 600, //!< (string)
        //601-620 reserved

        // Oela
        OelaEgoLaneProjectionX          = 700,
        OelaEgoLaneProjectionY          = 701,
        OelaLateralEgoLaneDistance      = 702,
        OelaLongitudinalEgoLaneDistance = 703,
        OelaEgoLaneFlag                 = 704,

        Oela_EgoLaneProjectionX          = OelaEgoLaneProjectionX, //!< Deprecated
        Oela_EgoLaneProjectionY          = OelaEgoLaneProjectionY, //!< Deprecated
        Oela_LateralEgoLaneDistance      = OelaLateralEgoLaneDistance, //!< Deprecated
        Oela_LongitudinalEgoLaneDistance = OelaLongitudinalEgoLaneDistance, //!< Deprecated
        Oela_EgoLaneFlag                 = OelaEgoLaneFlag, //!< Deprecated

        RearMonitoringFlag = 705, //!< (bit-flag)

        // Object-to-Lane Association
        OlaLaneId              = 710, //!< ID of the lane the object is in (uint64_t)
        OlaLateralLaneDistance = 711, //!< [m]; lateral distance of object rear center to lane center; (float)
        OlaLateralLaneDistanceUncertainty = 712, //!< [m]; uncertainty on lateral distance; (float)
        OlaLaneWidth                      = 713, //!< [m]; width of the lane the object is in; (float)
        OlaLaneWidthUncertainty           = 714, //!< [m]; uncertainty on the lane width; (float)
        OlaIsInEgoLane                    = 715, //!< is the lane the object is in the ego-lane?; (bool)

        Ola_LaneId                         = OlaLaneId, //!< Deprecated
        Ola_LateralLaneDistance            = OlaLateralLaneDistance, //!< Deprecated
        Ola_LateralLaneDistanceUncertainty = OlaLateralLaneDistanceUncertainty, //!< Deprecated
        Ola_LaneWidth                      = OlaLaneWidth, //!< Deprecated
        Ola_LaneWidthUncertainty           = OlaLaneWidthUncertainty, //!< Deprecated
        Ola_IsInEgoLane                    = OlaIsInEgoLane, //!< Deprecated
        //800-810 reserved
        //820 reserved

        //900-917 reserved

        //1051 reserved
        //1052 reserved
        //1053 reserved
        //1100 reserved
        //1101 reserved
        //1105 reserved
        //1106 reserved
        //1110 reserved

        // Hukseflux SR30 measurements
        Irradiance_compensated       = 1120,
        Irradiance_uncompensated     = 1121,
        Sensor_body_temperature      = 1122,
        Sensor_electrical_resistance = 1123,
        Sensor_voltage_output        = 1124,
        Humidity                     = 1125,
        Humidity_temperature         = 1126,
        Pressure                     = 1127,
        Pressure_average             = 1128,
        Pressure_temperature         = 1129,
        Pressure_temperature_average = 1130,
        // 1131-1139 reserved

        // Special CAN measurements
        WindshieldWipers = 1140,
        // 1150-1350 reserved

        // bot move plan
        EmergencyDecelerationRate = 1400
    }; // Value

public:
    explicit MeasurementKeyIn2821(const uint16_t key) : ComparableUnconvertable<uint16_t>(key) {}

    explicit MeasurementKeyIn2821(const Value key) : ComparableUnconvertable<uint16_t>(static_cast<uint16_t>(key)) {}

    MeasurementKeyIn2821(const MeasurementKeyIn2821& key)
      : ComparableUnconvertable<uint16_t>(static_cast<uint16_t>(key))
    {}

    MeasurementKeyIn2821() : ComparableUnconvertable<uint16_t>(static_cast<uint16_t>(Value::Undefined)) {}

    inline bool operator==(const Value& val) { return val == static_cast<Value>(m_data); }
    inline bool operator!=(const Value& val) { return val != static_cast<Value>(m_data); }

    MeasurementKeyIn2821& operator=(const MeasurementKeyIn2821& other);

public:
    bool isset() const { return (m_data != 0); }
    void unset() { m_data = 0; }

public:
    static std::streamsize getSerializedSize() { return sizeof(uint16_t); }

public:
    std::istream& readBE(std::istream& is)
    {
        microvision::common::sdk::readBE(is, m_data);
        return is;
    }

    std::ostream& writeBE(std::ostream& os) const
    {
        microvision::common::sdk::writeBE(os, m_data);
        return os;
    }

    void readBE(const char*& target) { microvision::common::sdk::readBE(target, m_data); }

    void writeBE(char*& target) const { microvision::common::sdk::writeBE(target, m_data); }

public: // friend functions for serialization
    template<typename TT>
    friend void readBE(std::istream& is, TT& value);
    template<typename TT>
    friend void writeBE(std::ostream& os, const TT& value);
}; // MeasurementKey

//==============================================================================

//==============================================================================
// Serialization
//==============================================================================

//==============================================================================
template<>
inline void readBE<MeasurementKeyIn2821>(std::istream& is, MeasurementKeyIn2821& k)
{
    microvision::common::sdk::readBE(is, k.m_data);
}

//==============================================================================
template<>
inline void writeBE<MeasurementKeyIn2821>(std::ostream& os, const MeasurementKeyIn2821& k)
{
    microvision::common::sdk::writeBE(os, k.m_data);
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace boost {
//==============================================================================

template<>
struct hash<microvision::common::sdk::MeasurementKeyIn2821>
{
    std::size_t operator()(microvision::common::sdk::MeasurementKeyIn2821 const& key) const
    {
        hash<uint16_t> h;
        return h(key);
    }
}; // :hash<DataTypeId>

//==============================================================================
} // namespace boost
//==============================================================================

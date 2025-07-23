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
//! \date Jan 18, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU vehicle state
//!
//! Vehicle state data available from FUSION SYSTEM and AppBase2 (ECU).
//!
//! All angles, position and distances are given in the ISO 8855 / DIN 70000 coordinate system.
//!
//! General data type: \ref microvision::common::sdk::VehicleState
//------------------------------------------------------------------------------
class VehicleState2808 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.vehiclestate2808"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    VehicleState2808();
    ~VehicleState2808() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Returns the reference point as WGS84 type. If reference point is not set,
    //!
    //! lat and lon will be 0
    //----------------------------------------
    PositionWgs84 getReferencePoint() const;

public:
    //!\brief flags to be used to signal vehicle state source
    enum class VehicleStateSource : uint16_t
    {
        Unknown              = 0x0000U, //!< unknown source
        Can                  = 0x0001U, //!< from CAN data
        Gps                  = 0x0002U, //!< GPS based
        Imu                  = 0x0004U, //!< calculated from IMU
        LidarOdometry        = 0x0008U, //!< based on lidar odometry
        LandmarkLocalication = 0x0010U, //!< landmark based estimation
        Imported             = 0x0100U //!< imported data, not self-estimated
    };

public: // getter
    uint32_t getMicrosecondsSinceStartup() const { return m_microsecondsSinceStartup; }
    NtpTime getTimestamp() const { return m_timestamp; }
    VehicleStateSource getSources() const { return m_sources; }
    uint16_t getBlindPredictionAge() const { return m_blindPredictionAge; }

    double getLatitude() const { return m_latitude; }
    float getLatitudeSigma() const { return m_latitudeSigma; }
    double getLongitude() const { return m_longitude; }
    float getLongitudeSigma() const { return m_longitudeSigma; }
    float getAltitude() const { return m_altitude; }
    float getAltitudeSigma() const { return m_altitudeSigma; }

    double getXPosition() const { return m_xPosition; }
    float getXPositionSigma() const { return m_xPositionSigma; }
    double getYPosition() const { return m_yPosition; }
    float getYPositionSigma() const { return m_yPositionSigma; }
    float getZPosition() const { return m_zPosition; }
    float getZPositionSigma() const { return m_zPositionSigma; }

    float getXyCorrelation() const { return m_xyCorrelation; }
    float getXzCorrelation() const { return m_xzCorrelation; }
    float getYzCorrelation() const { return m_yzCorrelation; }

    float getCourseAngle() const { return m_courseAngle; }
    float getCourseAngleSigma() const { return m_courseAngleSigma; }
    float getHeadingAngle() const { return m_headingAngle; }
    float getHeadingAngleSigma() const { return m_headingAngleSigma; }

    float getVehiclePitchAngle() const { return m_vehiclePitchAngle; }
    float getVehiclePitchAngleSigma() const { return m_vehiclePitchAngleSigma; }
    float getVehicleRollAngle() const { return m_vehicleRollAngle; }
    float getVehicleRollAngleSigma() const { return m_vehicleRollAngleSigma; }

    float getLongitudinalVelocity() const { return m_vehicleVelocity; }
    float getLongitudinalVelocitySigma() const { return m_vehicleVelocitySigma; }

    float getYawRate() const { return m_yawRate; }
    float getYawRateSigma() const { return m_yawRateSigma; }

    float getLongitudinalAcceleration() const { return m_longitudinalAcceleration; }
    float getLongitudinalAccelarationSigma() const { return m_longitudinalAccelerationSigma; }
    float getCrossAcceleration() const { return m_crossAcceleration; }
    float getCrossAccelerationSigma() const { return m_crossAccelerationSigma; }

    float getSteerAngle() const { return m_steerAngle; }
    float getSteeringWheelAngle() const { return m_steeringWheelAngle; }

    float getVehicleWidth() const { return m_vehicleWidth; }
    float getMinTurningCircle() const { return m_minTurningCircle; }

    float getVehicleFrontToFrontAxle() const { return m_vehicleFrontToFrontAxle; }
    float getFrontAxleToRearAxle() const { return m_frontAxleToRearAxle; }
    float getRearAxleToVehicleRear() const { return m_rearAxleToVehicleRear; }
    float getRearAxleToOrigin() const { return m_rearAxleToOrigin; }
    float getSteerRatioCoeff0() const { return m_steerRatioCoeff0; }
    float getSteerRatioCoeff1() const { return m_steerRatioCoeff1; }
    float getSteerRatioCoeff2() const { return m_steerRatioCoeff2; }
    float getSteerRatioCoeff3() const { return m_steerRatioCoeff3; }

public: // setter
    void setMicrosecondsSinceStartup(const uint32_t microseconds) { m_microsecondsSinceStartup = microseconds; }
    void setTimestamp(const NtpTime timestamp) { m_timestamp = timestamp; }
    void setSources(const VehicleStateSource sources) { m_sources = sources; }
    void setBlindPredictionAge(const uint16_t blindPredictionAge) { m_blindPredictionAge = blindPredictionAge; }

    void setLatitude(const double latitude) { m_latitude = latitude; }
    void setLatitudeSigma(const float latitudeSigma) { m_latitudeSigma = latitudeSigma; }
    void setLongitude(const double longitude) { m_longitude = longitude; }
    void setLongitudeSigma(const float longitudeSigma) { m_longitudeSigma = longitudeSigma; }
    void setAltitude(const float altitude) { m_altitude = altitude; }
    void setAltitudeSigma(const float altitudeSigma) { m_altitudeSigma = altitudeSigma; }

    void setXPosition(const double xPosition) { m_xPosition = xPosition; }
    void setXPositionSigma(const float xPositionSigma) { m_xPositionSigma = xPositionSigma; }
    void setYPosition(const double yPosition) { m_yPosition = yPosition; }
    void setYPositionSigma(const float yPositionSigma) { m_yPositionSigma = yPositionSigma; }
    void setZPosition(const float zPosition) { m_zPosition = zPosition; }
    void setZPositionSigma(const float zPositionSigma) { m_zPositionSigma = zPositionSigma; }

    void setXyCorrelation(const float xyCorrelation) { m_xyCorrelation = xyCorrelation; }
    void setXzCorrelation(const float xzCorrelation) { m_xzCorrelation = xzCorrelation; }
    void setYzCorrelation(const float yzCorrelation) { m_yzCorrelation = yzCorrelation; }

    void setCourseAngle(const float courseAngle) { m_courseAngle = courseAngle; }
    void setCourseAngleSigma(const float courseAngleSigma) { m_courseAngleSigma = courseAngleSigma; }
    void setHeadingAngle(const float headingAngle) { m_headingAngle = headingAngle; }
    void setHeadingAngleSigma(const float headingAngleSigma) { m_headingAngleSigma = headingAngleSigma; }

    void setVehiclePitchAngle(const float vehiclePitchAngle) { m_vehiclePitchAngle = vehiclePitchAngle; }
    void setVehiclePitchAngleSigma(const float vehiclePitchAngleSigma)
    {
        m_vehiclePitchAngleSigma = vehiclePitchAngleSigma;
    }
    void setVehicleRollAngle(const float vehicleRollAngle) { m_vehicleRollAngle = vehicleRollAngle; }
    void setVehicleRollAngleSigma(const float vehicleRollAngleSigma)
    {
        m_vehicleRollAngleSigma = vehicleRollAngleSigma;
    }

    void setLongitudinalVelocity(const float vehicleVelocity) { m_vehicleVelocity = vehicleVelocity; }
    void setLongitudinalVelocitySigma(const float vehicleVelocitySigma)
    {
        m_vehicleVelocitySigma = vehicleVelocitySigma;
    }

    void setYawRate(const float yawRate) { m_yawRate = yawRate; }
    void setYawRateSigma(const float yawRateSigma) { m_yawRateSigma = yawRateSigma; }

    void setLongitudinalAcceleration(const float longitudinalAcceleration)
    {
        m_longitudinalAcceleration = longitudinalAcceleration;
    }
    void setLongitudinalAccelerationSigma(const float longitudinalAccelerationSigma)
    {
        m_longitudinalAccelerationSigma = longitudinalAccelerationSigma;
    }
    void setCrossAcceleration(const float crossAcceleration) { m_crossAcceleration = crossAcceleration; }
    void setCrossAccelerationSigma(const float crossAccelerationSigma)
    {
        m_crossAccelerationSigma = crossAccelerationSigma;
    }

    void setSteerAngle(const float steerAngle) { m_steerAngle = steerAngle; }
    void setSteeringWheelAngle(const float steeringWheelAngle) { m_steeringWheelAngle = steeringWheelAngle; }

    void setVehicleWidth(const float vehicleWidth) { m_vehicleWidth = vehicleWidth; }
    void setMinTurningCircle(const float minTurningCircle) { m_minTurningCircle = minTurningCircle; }

    void setVehicleFrontToFrontAxle(const float vehicleFrontToFrontAxle)
    {
        m_vehicleFrontToFrontAxle = vehicleFrontToFrontAxle;
    }
    void setFrontAxleToRearAxle(const float frontAxleToRearAxle) { m_frontAxleToRearAxle = frontAxleToRearAxle; }
    void setRearAxleToVehicleRear(const float rearAxleToVehicleRear)
    {
        m_rearAxleToVehicleRear = rearAxleToVehicleRear;
    }
    void setRearAxleToOrigin(const float rearAxleToOrigin) { m_rearAxleToOrigin = rearAxleToOrigin; }
    void setSteerRatioCoeff0(const float steerRatioCoeff0) { m_steerRatioCoeff0 = steerRatioCoeff0; }
    void setSteerRatioCoeff1(const float steerRatioCoeff1) { m_steerRatioCoeff1 = steerRatioCoeff1; }
    void setSteerRatioCoeff2(const float steerRatioCoeff2) { m_steerRatioCoeff2 = steerRatioCoeff2; }
    void setSteerRatioCoeff3(const float steerRatioCoeff3) { m_steerRatioCoeff3 = steerRatioCoeff3; }

protected:
    uint32_t m_microsecondsSinceStartup{0}; //!< microseconds since startup.uint32_t m_microseconds{0};
    NtpTime m_timestamp{}; //!< timestamp of this data
    VehicleStateSource m_sources{VehicleStateSource::Unknown};
    uint16_t m_blindPredictionAge{0};

    float m_altitude{NaN}; //!< initial global Position(origin)[m]
    float m_altitudeSigma{NaN}; //!< [m]
    double m_longitude{NaN_double}; //!< initial global Position(origin)[deg]
    float m_longitudeSigma{NaN}; //!< [deg]
    double m_latitude{NaN_double}; //!< initial global Position(origin)[deg]
    float m_latitudeSigma{NaN}; //!< [deg]

    double m_xPosition{NaN_double}; //!< Absolute X Position from origin | [m]
    float m_xPositionSigma{NaN}; //!< standard deviation [m]
    double m_yPosition{NaN_double}; //!< Absolute X Position from origin | [m]
    float m_yPositionSigma{NaN}; //!< standard deviation [m]
    float m_zPosition{NaN}; //!< Absolute X Position from origin | [m]
    float m_zPositionSigma{NaN}; //!< standard deviation [m]

    float m_xyCorrelation{NaN}; //!< standard deviation [m]
    float m_xzCorrelation{NaN}; //!< standard deviation [m]
    float m_yzCorrelation{NaN}; //!< standard deviation [m]

    float m_courseAngle{NaN}; //!< Absolute orientation at time timeStamp | [rad]
    float m_courseAngleSigma{NaN}; //!< [rad]
    float m_headingAngle{NaN}; //!< heading angle [rad]
    float m_headingAngleSigma{NaN}; //!< standard deviation [rad]

    float m_vehiclePitchAngle{NaN}; //!< current vehicle pitch angle [rad]
    float m_vehiclePitchAngleSigma{NaN}; //!< standard deviation [rad]
    float m_vehicleRollAngle{NaN}; //!< current vehicle roll angle [rad]
    float m_vehicleRollAngleSigma{NaN}; //!< standard deviation [rad]

    float m_vehicleVelocity{NaN}; //!< Current longitudinal velocity of the vehicle | [m/s]
    float m_vehicleVelocitySigma{NaN}; //!< standard deviation velocity of vehicle [m/s]

    float m_yawRate{NaN}; //!< Difference in Heading during Timediff [rad/s]
    float m_yawRateSigma{NaN}; //!< [rad/s]

    float m_longitudinalAcceleration{NaN}; //!< Current longitudinal acceleration of the vehicle | [m/s^2]
    float m_longitudinalAccelerationSigma{NaN}; //!< standard deviation longitudinal acceleration of vehicle [m/s²]
    float m_crossAcceleration{NaN}; //!< current crossAcceleration of vehicle [m/s²]
    float m_crossAccelerationSigma{NaN}; //!< standard deviation crossAcceleration of vehicle [m/s²]

    float m_steerAngle{NaN}; //!< [rad]
    float m_steeringWheelAngle{NaN}; //!< steering wheel angle [rad]

    float m_vehicleWidth{NaN}; //!< Vehicle width [m]
    float m_minTurningCircle{NaN}; //!< [m]

    float m_vehicleFrontToFrontAxle{NaN}; //!< Distance: vehicle's front axle to vehicle's front [m]
    float m_frontAxleToRearAxle{NaN}; //!< Distance: vehicle's rear axle to vehicle's front axle [m]
    float m_rearAxleToVehicleRear{NaN}; //!< Distance: vehicle's rear axle to vehicle's rear [m]
    float m_rearAxleToOrigin{NaN}; //!< Distance: vehicle's rear axle to origin [m]
    float m_steerRatioCoeff0{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff1{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff2{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
    float m_steerRatioCoeff3{NaN}; //!< s_3*x^3 + s_2*x^2 + s-2*x^1 + s_0
}; // VehicleState2808

//==============================================================================

//==============================================================================

bool operator==(const VehicleState2808& lhs, const VehicleState2808& rhs);
bool operator!=(const VehicleState2808& lhs, const VehicleState2808& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

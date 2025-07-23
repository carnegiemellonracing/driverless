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
//! \date Mar 23, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImuSourceIn9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImuInsQualityIn9001.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief GpsImu
//!
//! This data type contains unprocessed position and motion data provided by a connected GPS/IMU device, e.g. XSens MTi-G.
//!
//! General data type: \ref microvision::common::sdk::GpsImu
//------------------------------------------------------------------------------
class GpsImu9001 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.gpsimu9001"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    GpsImu9001();
    ~GpsImu9001() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Get the latitude [rad].
    //!\return The latitude.
    //----------------------------------------
    double getLatitudeInRad() const { return m_latitudeInRad; }

    //========================================
    //!\brief Get the latitude uncertainty [m].
    //!\return The latitude uncertainty.
    //----------------------------------------
    double getLatitudeSigmaInMeter() const { return m_latitudeSigmaInMeter; }

    //========================================
    //!\brief Get the longitude [rad].
    //!\return The longitude.
    //----------------------------------------
    double getLongitudeInRad() const { return m_longitudeInRad; }

    //========================================
    //!\brief Get the longitude uncertainty [m].
    //!\return The longitude uncertainty.
    //----------------------------------------
    double getLongitudeSigmaInMeter() const { return m_longitudeSigmaInMeter; }

    //========================================
    //!\brief Get the altitude uncertainty [m].
    //!\return The altitude uncertainty.
    //----------------------------------------
    double getAltitudeInMeter() const { return m_altitudeInMeter; }

    //========================================
    //!\brief Get the altitude uncertainty [m].
    //!\return The altitude uncertainty.
    //----------------------------------------
    double getAltitudeSigmaInMeter() const { return m_altitudeSigmaInMeter; }

    //========================================
    //!\brief Get the course angle [rad].
    //!\return The course angle relative to North.
    //!\note Interval: [0;2*Pi] The course angle is the angle the vehicle is traveling to. If you drift, it is different
    //!      to the yaw angle, which is the direction of the vehicle is heading/looking at.
    //!\example 0.0 = North, pi/2 = West
    //----------------------------------------
    double getCourseAngleInRad() const { return m_courseAngleInRad; }

    //========================================
    //!\brief Get the course angle uncertainty [rad].
    //!\return The course angle uncertainty.
    //----------------------------------------
    double getCourseAngleSigmaInRad() const { return m_courseAngleSigmaInRad; }

    //========================================
    //!\brief Get the Yaw Angle in [rad]
    //!\return The Yaw Angle relative to North.
    //!\note Interval: [0;2*Pi]. The yaw angle is the angle the vehicle is heading/looking at. If you drift, it is
    //!      different to the course angle, which is the direction of travelling or the track angle.
    //!\example 0.0 = North, pi/2 = West
    //----------------------------------------
    double getYawAngleInRad() const { return m_yawAngleInRad; }

    //========================================
    //!\brief Set the yaw angle uncertainty [rad].
    //!\return The yaw angle uncertainty.
    //----------------------------------------
    double getYawAngleSigmaInRad() const { return m_yawAngleSigmaInRad; }

    //========================================
    //!\brief Get the pitch angle [rad].
    //!\return The pitch angle.
    //----------------------------------------
    double getPitchAngleInRad() const { return m_pitchAngleInRad; }

    //========================================
    //!\brief Get the pitch angle uncertainty [rad].
    //!\return The pitch angle uncertainty.
    //----------------------------------------
    double getPitchAngleSigmaInRad() const { return m_pitchAngleSigmaInRad; }

    //========================================
    //!\brief Get the roll angle [rad].
    //!\return The roll angle.
    //----------------------------------------
    double getRollAngleInRad() const { return m_rollAngleInRad; }

    //========================================
    //!\brief Get the roll angle uncertainty [rad].
    //!\return The roll angle uncertainty.
    //----------------------------------------
    double getRollAngleSigmaInRad() const { return m_rollAngleSigmaInRad; }

    //========================================
    //!\brief Get the cross angle [m/s^2].
    //!\return The cross angle.
    //----------------------------------------
    double getCrossAccelerationInMeterPerSecond2() const { return m_crossAccelerationInMeterPerSecond2; }

    //========================================
    //!\brief Get the cross angle uncertainty [m/s^2].
    //!\return The cross angle uncertainty.
    //----------------------------------------
    double getCrossAccelerationSigmaInMeterPerSecond2() const { return m_crossAccelerationSigmaInMeterPerSecond2; }

    //========================================
    //!\brief Get the longitudinal angle  [m/s^2].
    //!\return The longitudinal angle.
    //----------------------------------------
    double getLongitudinalAccelerationInMeterPerSecond2() const { return m_longitudinalAccelerationInMeterPerSecond2; }

    //========================================
    //!\brief Get the longitudinal angle uncertainty [m/s^2].
    //!\return The longitudinal angle uncertainty.
    //----------------------------------------
    double getLongitudinalAccelerationSigmaInMeterPerSecond2() const
    {
        return m_longitudinalAccelerationSigmaInMeterPerSecond2;
    }

    //========================================
    //!\brief Get the vertical angle [m/s^2].
    //!\return The vertical angle.
    //----------------------------------------
    double getVerticalAccelerationInMeterPerSecond2() const { return m_verticalAccelerationInMeterPerSecond2; }

    //========================================
    //!\brief Get the vertical angle uncertainty [m/s^2].
    //!\return The vertical angle uncertainty.
    //----------------------------------------
    double getVerticalAccelerationSigmaInMeterPerSecond2() const
    {
        return m_verticalAccelerationSigmaInMeterPerSecond2;
    }

    //========================================
    //!\brief Get the velocity north [m/s].
    //!\return The velocity north.
    //----------------------------------------
    double getVelocityNorthInMeterPerSecond() const { return m_velocityNorthInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity north uncertainty [m/s].
    //!\return The velocity north uncertainty.
    //----------------------------------------
    double getVelocityNorthSigmaInMeterPerSecond() const { return m_velocityNorthSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity west [m/s].
    //!\return The velocity west.
    //----------------------------------------
    double getVelocityWestInMeterPerSecond() const { return m_velocityWestInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity west uncertainty [m/s].
    //!\return The velocity west uncertainty.
    //----------------------------------------
    double getVelocityWestSigmaInMeterPerSecond() const { return m_velocityWestSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity up [m/s].
    //!\return The velocity up.
    //----------------------------------------
    double getVelocityUpInMeterPerSecond() const { return m_velocityUpInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity up uncertainty [m/s].
    //!\return The velocity up uncertainty.
    //----------------------------------------
    double getVelocityUpSigmaInMeterPerSecond() const { return m_velocityUpSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity in x direction [m/s].
    //!\return The velocity in x direction.
    //----------------------------------------
    double getVelocityXInMeterPerSecond() const { return m_velocityXInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity uncertainty in x direction [m/s].
    //!\return The velocity uncertainty in x direction.
    //----------------------------------------
    double getVelocityXSigmaInMeterPerSecond() const { return m_velocityXSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity in y direction [m/s].
    //!\return The velocity in y direction.
    //----------------------------------------
    double getVelocityYInMeterPerSecond() const { return m_velocityYInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity uncertainty in y direction [m/s].
    //!\return The velocity uncertainty in y direction.
    //----------------------------------------
    double getVelocityYSigmaInMeterPerSecond() const { return m_velocityYSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity in z direction [m/s].
    //!\return The velocity in z direction.
    //----------------------------------------
    double getVelocityZInMeterPerSecond() const { return m_velocityZInMeterPerSecond; }

    //========================================
    //!\brief Get the velocity uncertainty in z direction [m/s].
    //!\return The velocity uncertainty in z direction.
    //----------------------------------------
    double getVelocityZSigmaInMeterPerSecond() const { return m_velocityZSigmaInMeterPerSecond; }

    //========================================
    //!\brief Get the roll rate [rad/s].
    //!\return The roll rate.
    //----------------------------------------
    double getYawRateInRadPerSecond() const { return m_yawRateInRadPerSecond; }

    //========================================
    //!\brief Get the roll rate uncertainty [rad/s].
    //!\return The roll rate uncertainty.
    //----------------------------------------
    double getYawRateSigmaInRadPerSecond() const { return m_yawRateSigmaInRadPerSecond; }

    //========================================
    //!\brief Get the yaw rate [rad/s].
    //!\return The yaw rate.
    //----------------------------------------
    double getPitchRateInRadPerSecond() const { return m_pitchRateInRadPerSecond; }

    //========================================
    //!\brief Get the yaw rate uncertainty [rad/s].
    //!\return The yaw rate uncertainty.
    //----------------------------------------
    double getPitchRateSigmaInRadPerSecond() const { return m_pitchRateSigmaInRadPerSecond; }

    //========================================
    //!\brief Get the pitch rate [rad/s].
    //!\return The pitch rate.
    //----------------------------------------
    double getRollRateInRadPerSecond() const { return m_rollRateInRadPerSecond; }

    //========================================
    //!\brief Get the pitch rate uncertainty [rad/s].
    //!\return The pitch rate uncertainty.
    //----------------------------------------
    double getRollRateSigmaInRadPerSecond() const { return m_rollRateSigmaInRadPerSecond; }

    //========================================
    //!\brief Get the GPS status [none] (tbd).
    //!\return The GPS status.
    //----------------------------------------
    double getGpsStatus() const { return m_gpsStatus; }

    //========================================
    //!\brief Get the number of satellites.
    //!\return The number of satellites.
    //----------------------------------------
    uint8_t getNoOfSatellites() const { return m_noOfSatellites; }

    //==================================================
    //!\brief Get the dilution of precision in x direction.
    //!\return The dilution of precision in x direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionX() const { return m_gpsDilutionOfPrecisionX; }

    //========================================
    //!\brief Get the dilution of precision in y direction.
    //!\return The dilution of precision in y direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionY() const { return m_gpsDilutionOfPrecisionY; }

    //========================================
    //!\brief Get the dilution of precision in horizontal direction.
    //!\return The dilution of precision in horizontal direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionHorizontal() const { return m_gpsDilutionOfPrecisionHorizontal; }

    //========================================
    //!\brief Get the dilution of precision in vertical direction.
    //!\return The dilution of precision in vertical direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionVertical() const { return m_gpsDilutionOfPrecisionVertical; }

    //========================================
    //!\brief Get the dilution of precision in position.
    //!\return The dilution of precision in position.
    //----------------------------------------
    double getGpsDilutionOfPrecisionPosition() const { return m_gpsDilutionOfPrecisionPosition; }

    //========================================
    //!\brief Get the dilution of precision in time.
    //!\return The dilution of precision in time.
    //----------------------------------------
    double getGpsDilutionOfPrecisionTime() const { return m_gpsDilutionOfPrecisionTime; }

    //========================================
    //!\brief Get the dilution of precision in geometric.
    //!\return The dilution of precision in geometric.
    //----------------------------------------
    double getGpsDilutionOfPrecisionGeometric() const { return m_gpsDilutionOfPrecisionGeometric; }

    //========================================
    //!\brief Get the timestamp.
    //!\return The timestamp.
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_timestamp; }

    //========================================
    //!\brief Get the source of the GPS/IMU measurements.
    //!\return The source of the GPS/IMU measurements.
    //----------------------------------------
    const GpsImuSourceIn9001& getSource() const { return m_source; }

    //========================================
    //!\brief Get the quality flag of the GPS/IMU measurements.
    //!\return The quality flag of the GPS/IMU measurements.
    //----------------------------------------
    const GpsImuInsQualityIn9001& getInsQuality() const { return m_insQuality; }

public:
    //========================================
    //!\brief Set the latitude [rad].
    //!\param[in] latitude  The new latitude.
    //----------------------------------------
    void setLatitudeInRad(const double latitude) { m_latitudeInRad = latitude; }

    //========================================
    //!\brief Set the latitude uncertainty [m].
    //!\param[in] latitudeSigma The latitude uncertainty.
    //----------------------------------------
    void setLatitudeSigmaInMeter(const double latitudeSigma) { m_latitudeSigmaInMeter = latitudeSigma; }

    //========================================
    //!\brief Set the longitude [rad].
    //!\param[in] longitude The new longitude.
    //----------------------------------------
    void setLongitudeInRad(const double longitude) { m_longitudeInRad = longitude; }

    //========================================
    //!\brief Set the longitude uncertainty [m].
    //!\param[in] longitudeSigma  The longitude uncertainty.
    //----------------------------------------
    void setLongitudeSigmaInMeter(const double longitudeSigma) { m_longitudeSigmaInMeter = longitudeSigma; }

    //========================================
    //!\brief Set the altitude uncertainty [m].
    //!\param[in] altitude  The altitude uncertainty.
    //----------------------------------------
    void setAltitudeInMeter(const double altitude) { m_altitudeInMeter = altitude; }

    //========================================
    //!\brief Set the altitude uncertainty [m].
    //!\param[in] altitudeSigma  The altitude uncertainty.
    //----------------------------------------
    void setAltitudeSigmaInMeter(const double altitudeSigma) { m_altitudeSigmaInMeter = altitudeSigma; }

    //========================================
    //!\brief Set the course angle [rad].
    //!\param[in] courseAngle  The course angle.
    //!\note Relative to North. Interval: [0;2*Pi] The course angle is the angle the vehicle is traveling to. If you
    //!       drift, it is different to the yaw angle, which is the direction of the vehicle is heading/looking at.
    //!\example 0.0 = North, pi/2 = West
    //----------------------------------------
    void setCourseAngleInRad(const double courseAngle) { m_courseAngleInRad = courseAngle; }

    //========================================
    //!\brief Set the course angle uncertainty [rad].
    //!\param[in] courseAngleSigma  The course angle uncertainty.
    //----------------------------------------
    void setCourseAngleSigmaInRad(const double courseAngleSigma) { m_courseAngleSigmaInRad = courseAngleSigma; }

    //========================================
    //!\brief Set the yaw Angle in [rad].
    //!\param[in] yawAngle  The new yaw Angle.
    //!\note Interval [0;2*Pi]. The yaw angle is the angle the vehicle is heading/looking at. If you drift, it is
    //!      different to the course angle, which is the direction of travelling or the track angle.
    //!\example 0.0 = North, pi/2 = West
    //----------------------------------------
    void setYawAngleInRad(const double yawAngle) { m_yawAngleInRad = yawAngle; }

    //========================================
    //!\brief Set the yaw angle uncertainty [rad].
    //!\param[in] yawAngleSigma  The yaw angle uncertainty.
    //----------------------------------------
    void setYawAngleSigmaInRad(const double yawAngleSigma) { m_yawAngleSigmaInRad = yawAngleSigma; }

    //========================================
    //!\brief Set the pitch angle uncertainty [rad].
    //!\param[in] pitchAngle  The pitch angle uncertainty.
    //----------------------------------------
    void setPitchAngleInRad(const double pitchAngle) { m_pitchAngleInRad = pitchAngle; }

    //========================================
    //!\brief Set the pitch angle uncertainty [rad].
    //!\param[in] pitchAngleSigma  The pitch angle uncertainty.
    //----------------------------------------
    void setPitchAngleSigmaInRad(const double pitchAngleSigma) { m_pitchAngleSigmaInRad = pitchAngleSigma; }

    //========================================
    //!\brief Set the roll angle [rad].
    //!\param[in] rollAngle  The roll angle.
    //----------------------------------------
    void setRollAngleInRad(const double rollAngle) { m_rollAngleInRad = rollAngle; }

    //========================================
    //!\brief Set the roll angle uncertainty [rad].
    //!\param[in] rollAngleSigma  The roll angle uncertainty.
    //----------------------------------------
    void setRollAngleSigmaInRad(const double rollAngleSigma) { m_rollAngleSigmaInRad = rollAngleSigma; }

    //========================================
    //!\brief Set the cross angle [m/s^2].
    //!\param[in] crossAcceleration  The cross angle.
    //----------------------------------------
    void setCrossAccelerationInMeterPerSecond2(const double crossAcceleration)
    {
        m_crossAccelerationInMeterPerSecond2 = crossAcceleration;
    }

    //========================================
    //!\brief Set the cross angle uncertainty [m/s^2].
    //!\param[in] crossAccSigma  The cross angle uncertainty.
    //----------------------------------------
    void setCrossAccelerationSigmaInMeterPerSecond2(const double crossAccSigma)
    {
        m_crossAccelerationSigmaInMeterPerSecond2 = crossAccSigma;
    }

    //========================================
    //!\brief Set the longitudinal angle  [m/s^2].
    //!\param[in] longAcc  The longitudinal angle.
    //----------------------------------------
    void setLongitudinalAccelerationInMeterPerSecond2(const double longAcc)
    {
        m_longitudinalAccelerationInMeterPerSecond2 = longAcc;
    }

    //========================================
    //!\brief Set the longitudinal angle uncertainty [m/s^2].
    //!\param[in] longAccSigma  The longitudinal angle uncertainty.
    //----------------------------------------
    void setLongitudinalAccelerationSigmaInMeterPerSecond2(const double longAccSigma)
    {
        m_longitudinalAccelerationSigmaInMeterPerSecond2 = longAccSigma;
    }

    //========================================
    //!\brief Set the vertical angle [m/s^2].
    //!\param[in] vertAcc The vertical angle.
    //----------------------------------------
    void setVerticalAccelerationInMeterPerSecond2(const double vertAcc)
    {
        m_verticalAccelerationInMeterPerSecond2 = vertAcc;
    }

    //========================================
    //!\brief Set the vertical angle uncertainty [m/s^2].
    //!\param[in] vertAccSigma  The new deviation.
    //----------------------------------------
    void setVerticalAccelerationSigmaInMeterPerSecond2(const double vertAccSigma)
    {
        m_verticalAccelerationSigmaInMeterPerSecond2 = vertAccSigma;
    }

    //========================================
    //!\brief Set the velocity north [m/s].
    //!\param[in] velocityNorth The velocity north.
    //----------------------------------------
    void setVelocityNorthInMeterPerSecond(const double velocityNorth)
    {
        m_velocityNorthInMeterPerSecond = velocityNorth;
    }

    //========================================
    //!\brief Set the velocity north uncertainty [m/s].
    //!\param[in] velocityNorthSigma  The new deviation.
    //----------------------------------------
    void setVelocityNorthSigmaInMeterPerSecond(const double velocityNorthSigma)
    {
        m_velocityNorthSigmaInMeterPerSecond = velocityNorthSigma;
    }

    //========================================
    //!\brief Set the velocity west [m/s].
    //!\param[in] velocityWest  The velocity west.
    //----------------------------------------
    void setVelocityWestInMeterPerSecond(const double velocityWest) { m_velocityWestInMeterPerSecond = velocityWest; }

    //========================================
    //!\brief Set the velocity west uncertainty [m/s].
    //!\param[in] velocityWestSigma  The new deviation.
    //----------------------------------------
    void setVelocityWestSigmaInMeterPerSecond(const double velocityWestSigma)
    {
        m_velocityWestSigmaInMeterPerSecond = velocityWestSigma;
    }

    //========================================
    //!\brief Set the velocity up [m/s].
    //!\param[in] velocityUp  The velocity up.
    //----------------------------------------
    void setVelocityUpInMeterPerSecond(const double velocityUp) { m_velocityUpInMeterPerSecond = velocityUp; }

    //========================================
    //!\brief Set the velocity up uncertainty [m/s].
    //!\param[in] velocityUpSigma  The new deviation.
    //----------------------------------------
    void setVelocityUpSigmaInMeterPerSecond(const double velocityUpSigma)
    {
        m_velocityUpSigmaInMeterPerSecond = velocityUpSigma;
    }

    //========================================
    //!\brief Set the velocity in x direction [m/s].
    //!\param[in] velocityX  The velocity in x direction.
    //----------------------------------------
    void setVelocityXInMeterPerSecond(const double velocityX) { m_velocityXInMeterPerSecond = velocityX; }

    //========================================
    //!\brief Set the velocity uncertainty in x direction [m/s].
    //!\param[in] velocityXSigma  The new deviation.
    //----------------------------------------
    void setVelocityXSigmaInMeterPerSecond(const double velocityXSigma)
    {
        m_velocityXSigmaInMeterPerSecond = velocityXSigma;
    }

    //========================================
    //!\brief Set the velocity in y direction [m/s].
    //!\param[in] velocityY The velocity in y direction.
    //----------------------------------------
    void setVelocityYInMeterPerSecond(const double velocityY) { m_velocityYInMeterPerSecond = velocityY; }

    //========================================
    //!\brief Set the velocity uncertainty in y direction [m/s].
    //!\param[in] velocityYSigma  The new deviation.
    //----------------------------------------
    void setVelocityYSigmaInMeterPerSecond(const double velocityYSigma)
    {
        m_velocityYSigmaInMeterPerSecond = velocityYSigma;
    }

    //========================================
    //!\brief Set the velocity in z direction [m/s].
    //!\param[in] velocityZ  The velocity in z direction.
    //----------------------------------------
    void setVelocityZInMeterPerSecond(const double velocityZ) { m_velocityZInMeterPerSecond = velocityZ; }

    //========================================
    //!\brief Set the velocity uncertainty in z direction [m/s].
    //!\param[in] velocityZSigma  The new deviation.
    //----------------------------------------
    void setVelocityZSigmaInMeterPerSecond(const double velocityZSigma)
    {
        m_velocityZSigmaInMeterPerSecond = velocityZSigma;
    }

    //========================================
    //!\brief Set the roll rate [rad/s].
    //!\param[in] rollRate  The new roll rate.
    //----------------------------------------
    void setRollRateInRadPerSecond(const double rollRate) { m_rollRateInRadPerSecond = rollRate; }

    //========================================
    //!\brief Set the roll rate uncertainty [rad/s].
    //!\param[in] rollRateSigma  The new deviation.
    //----------------------------------------
    void setRollRateSigmaInRadPerSecond(const double rollRateSigma) { m_rollRateSigmaInRadPerSecond = rollRateSigma; }

    //========================================
    //!\brief Set the yaw rate [rad/s].
    //!\param[in] yawRate  The new yaw rate.
    //----------------------------------------
    void setYawRateInRadPerSecond(const double yawRate) { m_yawRateInRadPerSecond = yawRate; }

    //========================================
    //!\brief Set the yaw rate uncertainty [rad/s].
    //!\param[in] yawRateSigma  The new deviation.
    //----------------------------------------
    void setYawRateSigmaInRadPerSecond(const double yawRateSigma) { m_yawRateSigmaInRadPerSecond = yawRateSigma; }

    //========================================
    //!\brief Set the pitch rate [rad/s].
    //!\param[in] pitchRate The new pitch rate.
    //----------------------------------------
    void setPitchRateInRadPerSecond(const double pitchRate) { m_pitchRateInRadPerSecond = pitchRate; }

    //========================================
    //!\brief Set the pitch rate uncertainty [rad/s].
    //!\param[in] pitchRateSigma  The new deviation.
    //----------------------------------------
    void setPitchRateSigmaInRadPerSecond(const double pitchRateSigma)
    {
        m_pitchRateSigmaInRadPerSecond = pitchRateSigma;
    }

    //========================================
    //!\brief Set the GPS status [none] (tbd).
    //!\param[in] gpsStatus  The new Gps Status.
    //----------------------------------------
    void setGpsStatus(const double gpsStatus) { m_gpsStatus = gpsStatus; }

    //========================================
    //!\brief Set the number of satellites.
    //!\param[in] noOfSatellites  The new number of satellites.
    //----------------------------------------
    void setNoOfSatellites(const uint8_t noOfSatellites) { m_noOfSatellites = noOfSatellites; }

    //========================================
    //!\brief Set the dilution of precision in x direction.
    //!\param[in] gpsDilutionOfPrecisionX  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionX(const double gpsDilutionOfPrecisionX)
    {
        m_gpsDilutionOfPrecisionX = gpsDilutionOfPrecisionX;
    }

    //========================================
    //!\brief Set the dilution of precision in y direction.
    //!\param[in] gpsDilutionOfPrecisionY  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionY(const double gpsDilutionOfPrecisionY)
    {
        m_gpsDilutionOfPrecisionY = gpsDilutionOfPrecisionY;
    }

    //========================================
    //!\brief Set the dilution of precision in horizontal direction.
    //!\param[in] gpsDilutionOfPrecisionHorizontal  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionHorizontal(const double gpsDilutionOfPrecisionHorizontal)
    {
        m_gpsDilutionOfPrecisionHorizontal = gpsDilutionOfPrecisionHorizontal;
    }

    //========================================
    //!\brief Set the dilution of precision in vertical direction.
    //!\param[in] gpsDilutionOfPrecisionVertical  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionVertical(const double gpsDilutionOfPrecisionVertical)
    {
        m_gpsDilutionOfPrecisionVertical = gpsDilutionOfPrecisionVertical;
    }

    //========================================
    //!\brief Set the dilution of precision in position.
    //!\param[in] gpsDilutionOfPrecisionPosition  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionPosition(const double gpsDilutionOfPrecisionPosition)
    {
        m_gpsDilutionOfPrecisionPosition = gpsDilutionOfPrecisionPosition;
    }

    //========================================
    //!\brief Set the dilution of precision in time.
    //!\param[in] gpsDilutionOfPrecisionTime  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionTime(const double gpsDilutionOfPrecisionTime)
    {
        m_gpsDilutionOfPrecisionTime = gpsDilutionOfPrecisionTime;
    }

    //========================================
    //!\brief Set the dilution of precision in geometric.
    //!\param[in] gpsDilutionOfPrecisionGeometric  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionGeometric(const double gpsDilutionOfPrecisionGeometric)
    {
        m_gpsDilutionOfPrecisionGeometric = gpsDilutionOfPrecisionGeometric;
    }

    //========================================
    //!\brief Set the timestamp.
    //!\param[in] timestamp  The new timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_timestamp = timestamp; }

    //========================================
    //!\brief Set the source of the GPS/IMU measurements.
    //!\param[in] source  The source of the GPS/IMU measurements.
    //----------------------------------------
    void setSource(const GpsImuSourceIn9001 source) { m_source = source; }

    //========================================
    //!\brief Get the quality flag of the GPS/IMU measurements.
    //!\param[in] insQuality  The new quality flag of the GPS/IMU measurements.
    //----------------------------------------
    void setInsQuality(const GpsImuInsQualityIn9001 insQuality) { m_insQuality = insQuality; }

protected:
    double m_latitudeInRad{NaN_double}; //!< The latitude in [rad].
    double m_latitudeSigmaInMeter{NaN_double}; //!< The latitude uncertainty in [m].
    double m_longitudeInRad{NaN_double}; //!< The longitude in [rad].
    double m_longitudeSigmaInMeter{NaN_double}; //!< The longitude uncertainty in [m].
    double m_altitudeInMeter{NaN_double}; //!< The altitude uncertainty in [m].
    double m_altitudeSigmaInMeter{NaN_double}; //!< The altitude uncertainty in [m].
    double m_courseAngleInRad{NaN_double}; //!< The course angle relative to North in [rad].
    double m_courseAngleSigmaInRad{NaN_double}; //!< The course angle uncertainty in [rad].
    double m_yawAngleInRad{NaN_double}; //!< The Yaw Angle relative to North in [rad].
    double m_yawAngleSigmaInRad{NaN_double}; //!< The yaw angle uncertainty in [rad].
    double m_pitchAngleInRad{NaN_double}; //!< The pitch angle in [rad].
    double m_pitchAngleSigmaInRad{NaN_double}; //!< The pitch angle uncertainty in [rad].
    double m_rollAngleInRad{NaN_double}; //!< The roll angle in [rad].
    double m_rollAngleSigmaInRad{NaN_double}; //!< The roll angle uncertainty in [rad].

    double m_crossAccelerationInMeterPerSecond2{NaN_double}; //!< The cross angle in [m/s^2].
    double m_crossAccelerationSigmaInMeterPerSecond2{NaN_double}; //!< The cross angle uncertainty in [m/s^2].
    double m_longitudinalAccelerationInMeterPerSecond2{NaN_double}; //!< The longitudinal angle in [m/s^2].
    double m_longitudinalAccelerationSigmaInMeterPerSecond2{
        NaN_double}; //!< The longitudinal angle uncertainty in [m/s^2].
    double m_verticalAccelerationInMeterPerSecond2{NaN_double}; //!< The vertical angle in [m/s^2].
    double m_verticalAccelerationSigmaInMeterPerSecond2{NaN_double}; //!< The vertical angle uncertainty in [m/s^2].
    double m_velocityNorthInMeterPerSecond{NaN_double}; //!< The velocity north in [m/s].
    double m_velocityNorthSigmaInMeterPerSecond{NaN_double}; //!< The velocity north uncertainty in [m/s].
    double m_velocityWestInMeterPerSecond{NaN_double}; //!< The velocity west in [m/s].
    double m_velocityWestSigmaInMeterPerSecond{NaN_double}; //!< The velocity west uncertainty in [m/s].
    double m_velocityUpInMeterPerSecond{NaN_double}; //!< The velocity up in [m/s].
    double m_velocityUpSigmaInMeterPerSecond{NaN_double}; //!< The velocity up uncertainty in [m/s].
    double m_velocityXInMeterPerSecond{NaN_double}; //!< The velocity in x direction in [m/s].
    double m_velocityXSigmaInMeterPerSecond{NaN_double}; //!< The velocity uncertainty in x direction in [m/s].
    double m_velocityYInMeterPerSecond{NaN_double}; //!< The velocity in y direction in [m/s].
    double m_velocityYSigmaInMeterPerSecond{NaN_double}; //!< The velocity uncertainty in y direction in [m/s].
    double m_velocityZInMeterPerSecond{NaN_double}; //!< The velocity in z direction in [m/s].
    double m_velocityZSigmaInMeterPerSecond{NaN_double}; //!< The velocity uncertainty in z direction in [m/s].

    double m_rollRateInRadPerSecond{NaN_double}; //!< The roll rate in [rad/s].
    double m_rollRateSigmaInRadPerSecond{NaN_double}; //!< The roll rate uncertainty in [rad/s].
    double m_yawRateInRadPerSecond{NaN_double}; //!< The yaw rate in [rad/s].
    double m_yawRateSigmaInRadPerSecond{NaN_double}; //!< The yaw rate uncertainty in [rad/s].
    double m_pitchRateInRadPerSecond{NaN_double}; //!< The pitch rate in [rad/s].
    double m_pitchRateSigmaInRadPerSecond{NaN_double}; //!< The pitch rate uncertainty in [rad/s].

    double m_gpsStatus{NaN_double}; //!< The GPS status.
    uint8_t m_noOfSatellites{0}; //!< The number of satellites.

    double m_gpsDilutionOfPrecisionX{NaN_double}; //!< The dilution of precision in x direction.
    double m_gpsDilutionOfPrecisionY{NaN_double}; //!< The dilution of precision in y direction.
    double m_gpsDilutionOfPrecisionHorizontal{NaN_double}; //!< The dilution of precision in horizontal direction.
    double m_gpsDilutionOfPrecisionVertical{NaN_double}; //!< The dilution of precision in vertical direction.
    double m_gpsDilutionOfPrecisionPosition{NaN_double}; //!< The dilution of precision in position.
    double m_gpsDilutionOfPrecisionTime{NaN_double}; //!< The dilution of precision in time.
    double m_gpsDilutionOfPrecisionGeometric{NaN_double}; //!< The dilution of precision in geometric.

    Timestamp m_timestamp{}; //!< The timestamp.

    GpsImuSourceIn9001 m_source{GpsImuSourceIn9001::Unknown}; //!< The source of the GPS/IMU measurements.
    GpsImuInsQualityIn9001 m_insQuality{GpsImuInsQualityIn9001::Gps}; //!< The quality flag of the GPS/IMU measurements.

}; // GpsImu9001

//==============================================================================

bool operator==(const GpsImu9001& lhs, const GpsImu9001& rhs);
bool operator!=(const GpsImu9001& lhs, const GpsImu9001& rhs);

//==============================================================================

template<>
void writeBE<GpsImuSourceIn9001>(std::ostream& os, const GpsImuSourceIn9001& s);

template<>
void writeBE<GpsImuInsQualityIn9001>(std::ostream& os, const GpsImuInsQualityIn9001& s);

template<>
void readBE<GpsImuSourceIn9001>(std::istream& is, GpsImuSourceIn9001& tc);

template<>
void readBE<GpsImuInsQualityIn9001>(std::istream& is, GpsImuInsQualityIn9001& tc);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

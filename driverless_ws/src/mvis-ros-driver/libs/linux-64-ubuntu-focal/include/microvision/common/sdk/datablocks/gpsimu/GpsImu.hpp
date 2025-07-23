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
//! \date Sep 03, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/Timestamp.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImuSourceIn9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImuInsQualityIn9001.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9004.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

using GpsImuSource     = GpsImuSourceIn9001;
using GpsImuInsQuality = GpsImuInsQualityIn9001;

//==============================================================================

//==============================================================================
//! \brief GPS IMU
//!
//! Special data types:
//! \ref microvision::common::sdk::GpsImu9001
//! \ref microvision::common::sdk::GpsImu9004
//------------------------------------------------------------------------------
class GpsImu final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const GpsImu& lhs, const GpsImu& rhs);

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.gpsimu"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    GpsImu();
    ~GpsImu() override = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //!\brief Convert the source enum to a string.
    //!\param[in] s  The Source item to be converted
    //!              into a string.
    //!\return The Source item as string.
    //----------------------------------------
    static std::string toString(const GpsImuSource s);

public:
    //========================================
    //!\brief Get the latitude [rad].
    //----------------------------------------
    double getLatitudeInRad() const { return m_delegate.getLatitudeInRad(); }

    //========================================
    //!\brief Get the latitude uncertainty [m].
    //----------------------------------------
    double getLatitudeSigmaInMeter() const { return m_delegate.getLatitudeSigmaInMeter(); }

    //========================================
    //!\brief Get the longitude [rad].
    //----------------------------------------
    double getLongitudeInRad() const { return m_delegate.getLongitudeInRad(); }

    //========================================
    //!\brief Get the longitude uncertainty [m].
    //----------------------------------------
    double getLongitudeSigmaInMeter() const { return m_delegate.getLongitudeSigmaInMeter(); }

    //========================================
    //!\brief Get the altitude uncertainty [m].
    //----------------------------------------
    double getAltitudeInMeter() const { return m_delegate.getAltitudeInMeter(); }

    //========================================
    //!\brief Get the altitude uncertainty [m].
    //----------------------------------------
    double getAltitudeSigmaInMeter() const { return m_delegate.getAltitudeSigmaInMeter(); }

    //========================================
    //!\brief Get the course angle [rad].
    //!
    //! Relative to North. Interval: [0;2*Pi] The course angle is the angle the
    //! vehicle is traveling to. If you drift, it is different to the yaw angle,
    //! which is the direction of the vehicle is heading/looking at.
    //! example 0.0 = North, pi/2 = West
    //----------------------------------------
    double getCourseAngleInRad() const { return m_delegate.getCourseAngleInRad(); }

    //========================================
    //!\brief Get the course angle uncertainty [rad].
    //----------------------------------------
    double getCourseAngleSigmaInRad() const { return m_delegate.getCourseAngleSigmaInRad(); }

    //========================================
    //!\brief
    //! Yaw Angle in [rad], Interval [0;2*Pi]. The yaw angle is the angle the vehicle
    //! is heading/looking at. If you drift, it is different to the course angle,
    //! which is the direction of travelling or the track angle.
    //! example  0.0 = North, pi/2 = West
    //----------------------------------------
    double getYawAngleInRad() const { return m_delegate.getYawAngleInRad(); }

    //========================================
    //!\brief Set the yaw angle uncertainty [rad].
    //----------------------------------------
    double getYawAngleSigmaInRad() const { return m_delegate.getYawAngleSigmaInRad(); }

    //========================================
    //!\brief Get the pitch angle uncertainty [rad].
    //----------------------------------------
    double getPitchAngleInRad() const { return m_delegate.getPitchAngleInRad(); }

    //========================================
    //!\brief Get the pitch angle uncertainty [rad].
    //----------------------------------------
    double getPitchAngleSigmaInRad() const { return m_delegate.getPitchAngleSigmaInRad(); }

    //========================================
    //!\brief Get the roll angle [rad].
    //----------------------------------------
    double getRollAngleInRad() const { return m_delegate.getRollAngleInRad(); }

    //========================================
    //!\brief Get the roll angle uncertainty [rad].
    //----------------------------------------
    double getRollAngleSigmaInRad() const { return m_delegate.getRollAngleSigmaInRad(); }

    //========================================
    //!\brief Get the cross angle [m/s^2].
    //----------------------------------------
    double getCrossAccelerationInMeterPerSecond2() const { return m_delegate.getCrossAccelerationInMeterPerSecond2(); }

    //========================================
    //!\brief Get the cross angle uncertainty [m/s^2].
    //----------------------------------------
    double getCrossAccelerationSigmaInMeterPerSecond2() const
    {
        return m_delegate.getCrossAccelerationSigmaInMeterPerSecond2();
    }

    //========================================
    //!\brief Get the longitudinal angle  [m/s^2].
    //----------------------------------------
    double getLongitudinalAccelerationInMeterPerSecond2() const
    {
        return m_delegate.getLongitudinalAccelerationInMeterPerSecond2();
    }

    //========================================
    //!\brief Get the longitudinal angle uncertainty [m/s^2].
    //----------------------------------------
    double getLongitudinalAccelerationSigmaInMeterPerSecond2() const
    {
        return m_delegate.getLongitudinalAccelerationSigmaInMeterPerSecond2();
    }

    //========================================
    //!\brief Get the vertical angle [m/s^2].
    //----------------------------------------
    double getVerticalAccelerationInMeterPerSecond2() const
    {
        return m_delegate.getVerticalAccelerationInMeterPerSecond2();
    }

    //========================================
    //!\brief Get the vertical angle uncertainty [m/s^2].
    //----------------------------------------
    double getVerticalAccelerationSigmaInMeterPerSecond2() const
    {
        return m_delegate.getVerticalAccelerationSigmaInMeterPerSecond2();
    }

    //========================================
    //!\brief Get the velocity north [m/s].
    //----------------------------------------
    double getVelocityNorthInMeterPerSecond() const { return m_delegate.getVelocityNorthInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity north uncertainty [m/s].
    //----------------------------------------
    double getVelocityNorthSigmaInMeterPerSecond() const { return m_delegate.getVelocityNorthSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity west [m/s].
    //----------------------------------------
    double getVelocityWestInMeterPerSecond() const { return m_delegate.getVelocityWestInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity west uncertainty [m/s].
    //----------------------------------------
    double getVelocityWestSigmaInMeterPerSecond() const { return m_delegate.getVelocityWestSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity up [m/s].
    //----------------------------------------
    double getVelocityUpInMeterPerSecond() const { return m_delegate.getVelocityUpInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity up uncertainty [m/s].
    //----------------------------------------
    double getVelocityUpSigmaInMeterPerSecond() const { return m_delegate.getVelocityUpSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity in x direction [m/s].
    //----------------------------------------
    double getVelocityXInMeterPerSecond() const { return m_delegate.getVelocityXInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity uncertainty in x direction [m/s].
    //----------------------------------------
    double getVelocityXSigmaInMeterPerSecond() const { return m_delegate.getVelocityXSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity in y direction [m/s].
    //----------------------------------------
    double getVelocityYInMeterPerSecond() const { return m_delegate.getVelocityYInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity uncertainty in y direction [m/s].
    //----------------------------------------
    double getVelocityYSigmaInMeterPerSecond() const { return m_delegate.getVelocityYSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity in z direction [m/s].
    //----------------------------------------
    double getVelocityZInMeterPerSecond() const { return m_delegate.getVelocityZInMeterPerSecond(); }

    //========================================
    //!\brief Get the velocity uncertainty in z direction [m/s].
    //----------------------------------------
    double getVelocityZSigmaInMeterPerSecond() const { return m_delegate.getVelocityZSigmaInMeterPerSecond(); }

    //========================================
    //!\brief Get the roll rate [rad/s].
    //----------------------------------------
    double getYawRateInRadPerSecond() const { return m_delegate.getYawRateInRadPerSecond(); }

    //========================================
    //!\brief Get the roll rate uncertainty [rad/s].
    //----------------------------------------
    double getYawRateSigmaInRadPerSecond() const { return m_delegate.getYawRateSigmaInRadPerSecond(); }

    //========================================
    //!\brief Get the yaw rate [rad/s].
    //----------------------------------------
    double getPitchRateInRadPerSecond() const { return m_delegate.getPitchRateInRadPerSecond(); }

    //========================================
    //!\brief Get the yaw rate uncertainty [rad/s].
    //----------------------------------------
    double getPitchRateSigmaInRadPerSecond() const { return m_delegate.getPitchRateSigmaInRadPerSecond(); }

    //========================================
    //!\brief Get the pitch rate [rad/s].
    //----------------------------------------
    double getRollRateInRadPerSecond() const { return m_delegate.getRollRateInRadPerSecond(); }

    //========================================
    //!\brief Get the pitch rate uncertainty [rad/s].
    //----------------------------------------
    double getRollRateSigmaInRadPerSecond() const { return m_delegate.getRollRateSigmaInRadPerSecond(); }

    //========================================
    //!\brief Get the GPS status [none] (tbd).
    //----------------------------------------
    double getGpsStatus() const { return m_delegate.getGpsStatus(); }

    //========================================
    //!\brief Get the number of satellites.
    //----------------------------------------
    uint8_t getNoOfSatellites() const { return m_delegate.getNoOfSatellites(); }

    //==================================================
    //!\brief Get the dilution of precision in x direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionX() const { return m_delegate.getGpsDilutionOfPrecisionX(); }

    //========================================
    //!\brief Get the dilution of precision in y direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionY() const { return m_delegate.getGpsDilutionOfPrecisionY(); }

    //========================================
    //!\brief Get the dilution of precision in z direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionHorizontal() const { return m_delegate.getGpsDilutionOfPrecisionHorizontal(); }

    //========================================
    //!\brief Get the dilution of precision in vertical direction.
    //----------------------------------------
    double getGpsDilutionOfPrecisionVertical() const { return m_delegate.getGpsDilutionOfPrecisionVertical(); }

    //========================================
    //!\brief Get the dilution of precision in position.
    //----------------------------------------
    double getGpsDilutionOfPrecisionPosition() const { return m_delegate.getGpsDilutionOfPrecisionPosition(); }

    //========================================
    //!\brief Get the dilution of precision in time.
    //----------------------------------------
    double getGpsDilutionOfPrecisionTime() const { return m_delegate.getGpsDilutionOfPrecisionTime(); }

    //========================================
    //!\brief Get the dilution of precision in geometric.
    //----------------------------------------
    double getGpsDilutionOfPrecisionGeometric() const { return m_delegate.getGpsDilutionOfPrecisionGeometric(); }

    //========================================
    //!\brief Get timestamp.
    //----------------------------------------
    const Timestamp& getTimestamp() const { return m_delegate.getTimestamp(); }

    //========================================
    //!\brief Get the source of the GPS/IMU measurements.
    //----------------------------------------
    const GpsImuSource& getSource() const { return m_delegate.getSource(); }

    //========================================
    //!\brief Get the quality flag of the GPS/IMU measurements.
    //----------------------------------------
    const GpsImuInsQuality& getInsQuality() const { return m_delegate.getInsQuality(); }

    //========================================
    //!\brief Get the mounting yaw angle of the GPS/IMU device.
    //!\return The mounting yaw in [rad], between[-pi;pi].
    //----------------------------------------
    double getMountingAngleYaw() const { return m_delegate.getMountingAngleYaw(); }

    //========================================
    //!\brief Get the mounting pitch angle of the GPS/IMU device.
    //!\return The mounting pitch in [rad], between[-pi;pi].
    //----------------------------------------
    double getMountingAnglePitch() const { return m_delegate.getMountingAnglePitch(); }

    //========================================
    //!\brief Get the mounting roll angle of the GPS/IMU device.
    //!\return The mounting roll in [rad], between[-pi;pi].
    //----------------------------------------
    double getMountingAngleRoll() const { return m_delegate.getMountingAngleRoll(); }

    //========================================
    //!\brief Get the x mounting offset of the GPS/IMU device.
    //!\return The x mounting offset in [m].
    //----------------------------------------
    double getMountingOffsetX() const { return m_delegate.getMountingOffsetX(); }

    //========================================
    //!\brief Get the y mounting offset of the GPS/IMU device.
    //!\return The y mounting offset in [m].
    //----------------------------------------
    double getMountingOffsetY() const { return m_delegate.getMountingOffsetY(); }

    //========================================
    //!\brief Get the z mounting offset of the GPS/IMU device.
    //!\return The z mounting offset in [m].
    //----------------------------------------
    double getMountingOffsetZ() const { return m_delegate.getMountingOffsetZ(); }

public:
    //========================================
    //!\brief Set the latitude [rad].
    //----------------------------------------
    void setLatitudeInRad(const double latitude) { m_delegate.setLatitudeInRad(latitude); }

    //========================================
    //!\brief Set the latitude uncertainty [m].
    //----------------------------------------
    void setLatitudeSigmaInMeter(const double latitudeSigma) { m_delegate.setLatitudeSigmaInMeter(latitudeSigma); }

    //========================================
    //!\brief Set the longitude [rad].
    //----------------------------------------
    void setLongitudeInRad(const double longitude) { m_delegate.setLongitudeInRad(longitude); }

    //========================================
    //!\brief Set the longitude uncertainty [m].
    //----------------------------------------
    void setLongitudeSigmaInMeter(const double longitudeSigma) { m_delegate.setLongitudeSigmaInMeter(longitudeSigma); }

    //========================================
    //!\brief Set the altitude uncertainty [m].
    //----------------------------------------
    void setAltitudeInMeter(const double altitude) { m_delegate.setAltitudeInMeter(altitude); }

    //========================================
    //!\brief Set the altitude uncertainty [m].
    //----------------------------------------
    void setAltitudeSigmaInMeter(const double altitudeSigma) { m_delegate.setAltitudeSigmaInMeter(altitudeSigma); }

    //========================================
    //!\brief Set the course angle [rad].
    //!
    //! Relative to North. Interval: [0;2*Pi] The course angle is the angle the
    //! vehicle is traveling to. If you drift, it is different to the yaw angle,
    //! which is the direction of the vehicle is heading/looking at.
    //! example 0.0 = North, pi/2 = West
    //----------------------------------------
    void setCourseAngleInRad(const double courseAngle) { m_delegate.setCourseAngleInRad(courseAngle); }

    //========================================
    //!\brief Set the course angle uncertainty [rad].
    //----------------------------------------
    void setCourseAngleSigmaInRad(const double courseAngleSigma)
    {
        m_delegate.setCourseAngleSigmaInRad(courseAngleSigma);
    }

    //========================================
    //!\brief Set the yaw Angle in [rad].
    //!
    //! Interval [0;2*Pi]. The yaw angle is the angle the
    //! vehicle is heading/looking at. If you drift, it is different to the
    //! course angle, which is the direction of travelling or the track angle.
    //! example 0.0 = North, pi/2 = West
    //----------------------------------------
    void setYawAngleInRad(const double yawAngle) { m_delegate.setYawAngleInRad(yawAngle); }

    //========================================
    //!\brief Set the yaw angle uncertainty [rad].
    //----------------------------------------
    void setYawAngleSigmaInRad(const double yawAngleSigma) { m_delegate.setYawAngleSigmaInRad(yawAngleSigma); }

    //========================================
    //!\brief Set the pitch angle uncertainty [rad].
    //----------------------------------------
    void setPitchAngleInRad(const double pitchAngle) { m_delegate.setPitchAngleInRad(pitchAngle); }

    //========================================
    //!\brief Set the pitch angle uncertainty [rad].
    //----------------------------------------
    void setPitchAngleSigmaInRad(const double pitchAngleSigma) { m_delegate.setPitchAngleSigmaInRad(pitchAngleSigma); }

    //========================================
    //!\brief Set the roll angle [rad].
    //----------------------------------------
    void setRollAngleInRad(const double rollAngle) { m_delegate.setRollAngleInRad(rollAngle); }

    //========================================
    //!\brief Set the roll angle uncertainty [rad].
    //----------------------------------------
    void setRollAngleSigmaInRad(const double rollAngleSigma) { m_delegate.setRollAngleSigmaInRad(rollAngleSigma); }

    //========================================
    //!\brief Set the cross angle [m/s^2].
    //----------------------------------------
    void setCrossAccelerationInMeterPerSecond2(const double crossAcceleration)
    {
        m_delegate.setCrossAccelerationInMeterPerSecond2(crossAcceleration);
    }

    //========================================
    //!\brief Set the cross angle uncertainty [m/s^2].
    //----------------------------------------
    void setCrossAccelerationSigmaInMeterPerSecond2(const double crossAccSigma)
    {
        m_delegate.setCrossAccelerationSigmaInMeterPerSecond2(crossAccSigma);
    }

    //========================================
    //!\brief Set the longitudinal angle  [m/s^2].
    //----------------------------------------
    void setLongitudinalAccelerationInMeterPerSecond2(const double longAcc)
    {
        m_delegate.setLongitudinalAccelerationInMeterPerSecond2(longAcc);
    }

    //========================================
    //!\brief Set the longitudinal angle uncertainty [m/s^2].
    //----------------------------------------
    void setLongitudinalAccelerationSigmaInMeterPerSecond2(const double longAccSigma)
    {
        m_delegate.setLongitudinalAccelerationSigmaInMeterPerSecond2(longAccSigma);
    }

    //========================================
    //!\brief Set the vertical angle [m/s^2].
    //----------------------------------------
    void setVerticalAccelerationInMeterPerSecond2(const double vertAcc)
    {
        m_delegate.setVerticalAccelerationInMeterPerSecond2(vertAcc);
    }

    //========================================
    //!\brief Set the vertical angle uncertainty [m/s^2].
    //----------------------------------------
    void setVerticalAccelerationSigmaInMeterPerSecond2(const double vertAccSigma)
    {
        m_delegate.setVerticalAccelerationSigmaInMeterPerSecond2(vertAccSigma);
    }

    //========================================
    //!\brief Set the velocity north [m/s].
    //----------------------------------------
    void setVelocityNorthInMeterPerSecond(const double velocityNorth)
    {
        m_delegate.setVelocityNorthInMeterPerSecond(velocityNorth);
    }

    //========================================
    //!\brief Set the velocity north uncertainty [m/s].
    //----------------------------------------
    void setVelocityNorthSigmaInMeterPerSecond(const double velocityNorthSigma)
    {
        m_delegate.setVelocityNorthSigmaInMeterPerSecond(velocityNorthSigma);
    }

    //========================================
    //!\brief Set the velocity west [m/s].
    //----------------------------------------
    void setVelocityWestInMeterPerSecond(const double velocityWest)
    {
        m_delegate.setVelocityWestInMeterPerSecond(velocityWest);
    }

    //========================================
    //!\brief Set the velocity west uncertainty [m/s].
    //----------------------------------------
    void setVelocityWestSigmaInMeterPerSecond(const double velocityWestSigma)
    {
        m_delegate.setVelocityWestSigmaInMeterPerSecond(velocityWestSigma);
    }

    //========================================
    //!\brief Set the velocity up [m/s].
    //----------------------------------------
    void setVelocityUpInMeterPerSecond(const double velocityUp)
    {
        m_delegate.setVelocityUpInMeterPerSecond(velocityUp);
    }

    //========================================
    //!\brief Set the velocity up uncertainty [m/s].
    //----------------------------------------
    void setVelocityUpSigmaInMeterPerSecond(const double velocityUpSigma)
    {
        m_delegate.setVelocityUpSigmaInMeterPerSecond(velocityUpSigma);
    }

    //========================================
    //!\brief Set the velocity in x direction [m/s].
    //----------------------------------------
    void setVelocityXInMeterPerSecond(const double velocityX) { m_delegate.setVelocityXInMeterPerSecond(velocityX); }

    //========================================
    //!\brief Set the velocity uncertainty in x direction [m/s].
    //----------------------------------------
    void setVelocityXSigmaInMeterPerSecond(const double velocityXSigma)
    {
        m_delegate.setVelocityXSigmaInMeterPerSecond(velocityXSigma);
    }

    //========================================
    //!\brief Set the velocity in y direction [m/s].
    //----------------------------------------
    void setVelocityYInMeterPerSecond(const double velocityY) { m_delegate.setVelocityYInMeterPerSecond(velocityY); }

    //========================================
    //!\brief Set the velocity uncertainty in y direction [m/s].
    //----------------------------------------
    void setVelocityYSigmaInMeterPerSecond(const double velocityYSigma)
    {
        m_delegate.setVelocityYSigmaInMeterPerSecond(velocityYSigma);
    }

    //========================================
    //!\brief Set the velocity in z direction [m/s].
    //----------------------------------------
    void setVelocityZInMeterPerSecond(const double velocityZ) { m_delegate.setVelocityZInMeterPerSecond(velocityZ); }

    //========================================
    //!\brief Set the velocity uncertainty in z direction [m/s].
    //----------------------------------------
    void setVelocityZSigmaInMeterPerSecond(const double velocityZSigma)
    {
        m_delegate.setVelocityZSigmaInMeterPerSecond(velocityZSigma);
    }

    //========================================
    //!\brief Set the roll rate [rad/s].
    //----------------------------------------
    void setRollRateInRadPerSecond(const double rollRate) { m_delegate.setRollRateInRadPerSecond(rollRate); }

    //========================================
    //!\brief Set the roll rate uncertainty [rad/s].
    //----------------------------------------
    void setRollRateSigmaInRadPerSecond(const double rollRateSigma)
    {
        m_delegate.setRollRateSigmaInRadPerSecond(rollRateSigma);
    }

    //========================================
    //!\brief Set the yaw rate [rad/s].
    //----------------------------------------
    void setYawRateInRadPerSecond(const double yawRate) { m_delegate.setYawRateInRadPerSecond(yawRate); }

    //========================================
    //!\brief Set the yaw rate uncertainty [rad/s].
    //----------------------------------------
    void setYawRateSigmaInRadPerSecond(const double yawRateSigma)
    {
        m_delegate.setYawRateSigmaInRadPerSecond(yawRateSigma);
    }

    //========================================
    //!\brief Set the pitch rate [rad/s].
    //----------------------------------------
    void setPitchRateInRadPerSecond(const double pitchRate) { m_delegate.setPitchRateInRadPerSecond(pitchRate); }

    //========================================
    //!\brief Set the pitch rate uncertainty [rad/s].
    //----------------------------------------
    void setPitchRateSigmaInRadPerSecond(const double pitchRateSigma)
    {
        m_delegate.setPitchRateSigmaInRadPerSecond(pitchRateSigma);
    }

    //========================================
    //!\brief Set the GPS status [none] (tbd).
    //----------------------------------------
    void setGpsStatus(const double gpsStatus) { m_delegate.setGpsStatus(gpsStatus); }

    //========================================
    //!\brief Set the number of satellites.
    //----------------------------------------
    void setNoOfSatellites(const uint8_t noOfSatellites) { m_delegate.setNoOfSatellites(noOfSatellites); }

    //========================================
    //!\brief Set the dilution of precision in x direction.
    //----------------------------------------
    void setGpsDilutionOfPrecisionX(const double gpsDilutionOfPrecisionX)
    {
        m_delegate.setGpsDilutionOfPrecisionX(gpsDilutionOfPrecisionX);
    }

    //========================================
    //!\brief Set the dilution of precision in y direction.
    //----------------------------------------
    void setGpsDilutionOfPrecisionY(const double gpsDilutionOfPrecisionY)
    {
        m_delegate.setGpsDilutionOfPrecisionY(gpsDilutionOfPrecisionY);
    }

    //========================================
    //!\brief Set the dilution of precision in horizontal direction.
    //----------------------------------------
    void setGpsDilutionOfPrecisionHorizontal(const double gpsDilutionOfPrecisionHorizontal)
    {
        m_delegate.setGpsDilutionOfPrecisionHorizontal(gpsDilutionOfPrecisionHorizontal);
    }

    //========================================
    //!\brief Set the dilution of precision in vertical direction.
    //----------------------------------------
    void setGpsDilutionOfPrecisionVertical(const double gpsDilutionOfPrecisionVertical)
    {
        m_delegate.setGpsDilutionOfPrecisionVertical(gpsDilutionOfPrecisionVertical);
    }

    //========================================
    //!\brief Set the dilution of precision in position.
    //----------------------------------------
    void setGpsDilutionOfPrecisionPosition(const double gpsDilutionOfPrecisionPosition)
    {
        m_delegate.setGpsDilutionOfPrecisionPosition(gpsDilutionOfPrecisionPosition);
    }

    //========================================
    //!\brief Set the dilution of precision in time.
    //!\param[in] gpsDilutionOfPrecisionTime  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionTime(const double gpsDilutionOfPrecisionTime)
    {
        m_delegate.setGpsDilutionOfPrecisionTime(gpsDilutionOfPrecisionTime);
    }

    //========================================
    //!\brief Set the dilution of precision in geometric.
    //!\param[in] gpsDilutionOfPrecisionGeometric  The new dilution.
    //----------------------------------------
    void setGpsDilutionOfPrecisionGeometric(const double gpsDilutionOfPrecisionGeometric)
    {
        m_delegate.setGpsDilutionOfPrecisionGeometric(gpsDilutionOfPrecisionGeometric);
    }

    //========================================
    //!\brief Set the timestamp.
    //----------------------------------------
    void setTimestamp(const Timestamp& timestamp) { m_delegate.setTimestamp(timestamp); }

    //========================================
    //!\brief Set the source of the GPS/IMU measurements.
    //----------------------------------------
    void setSource(const GpsImuSource source) { m_delegate.setSource(source); }

    //========================================
    //!\brief Get the quality flag of the GPS/IMU measurements.
    //----------------------------------------
    void setInsQuality(const GpsImuInsQuality insQuality) { m_delegate.setInsQuality(insQuality); }

    //========================================
    //!\brief Set the mounting yaw angle of the GPS/IMU device.
    //!\param[in] yaw The mounting yaw in [rad], between[-pi;pi].
    //----------------------------------------
    void setMountingAngleYaw(double yaw) { m_delegate.setMountingAngleYaw(yaw); }

    //========================================
    //!\brief Set the mounting pitch angle of the GPS/IMU device.
    //!\param[in] pitch The mounting pitch in [rad], between[-pi;pi].
    //----------------------------------------
    void setMountingAnglePitch(double pitch) { m_delegate.setMountingAnglePitch(pitch); }

    //========================================
    //!\brief Set the mounting roll angle of the GPS/IMU device.
    //!\param[in] roll The mounting roll in [rad], between[-pi;pi].
    //----------------------------------------
    void setMountingAngleRoll(double roll) { m_delegate.setMountingAngleRoll(roll); }

    //========================================
    //!\brief Set the x mounting offset of the GPS/IMU device.
    //!\param[in] x The x mounting offset in [m].
    //----------------------------------------
    void setMountingOffsetX(double x) { m_delegate.setMountingOffsetX(x); }

    //========================================
    //!\brief Set the y mounting offset of the GPS/IMU device.
    //!\param[in] y The y mounting offset in [m].
    //----------------------------------------
    void setMountingOffsetY(double y) { m_delegate.setMountingOffsetY(y); }

    //========================================
    //!\brief Set the z mounting offset of the GPS/IMU device.
    //!\param[in] z The z mounting offset in [m].
    //----------------------------------------
    void setMountingOffsetZ(double z) { m_delegate.setMountingOffsetZ(z); }

protected:
    GpsImu9004 m_delegate;
}; // GpsImuContainer

//==============================================================================

bool operator==(const GpsImu& lhs, const GpsImu& rhs);
bool operator!=(const GpsImu& lhs, const GpsImu& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

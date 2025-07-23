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
//! \date Jan 24, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Math.hpp>
#include <microvision/common/sdk/NtpTime.hpp>

#include <microvision/common/sdk/misc/unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class EcefPoint;

//==============================================================================

class PositionWgs84 final
{
public:
    //========================================
    //! \brief  Sources that provide WGS-84 position information.
    //! \sa setSource()
    //----------------------------------------
    enum class SourceType : uint16_t
    {
        Unknown               = 0,
        GPS_SPS               = 1U,
        GPS_PPS               = 2U,
        GPS_SBAS              = 3U,
        GPS_SBAS_Omnistar_VBS = 4U,
        GPS_SBAS_Omnistar_HP  = 5U,
        GPS_GBAS              = 6U,
        GPS_GBAS_RTK_Float    = 7U,
        GPS_GBAS_RTK_Integer  = 8U,
        Imu                   = 9U,
        LandmarkPositioning   = 10U,
        Manual                = 11U,
        SourceType_Mission    = 12U,

        MaxSourceTypeId = 0xFFFFU
    };

public:
    static std::streamsize getSerializedSize_static();

public:
    static void llaToEcef(const double pointsLatitudeInRad, const double pointsLongitudeInRad, EcefPoint& pointsEcef);

    static void llaToEcef(const double pointsLatitudeInRad,
                          const double pointsLongitudeInRad,
                          const float pointsAltitudeInMeter,
                          EcefPoint& pointsEcef);

public:
    static void transformToTangentialPlane2d(const double originsLatitudeInRad,
                                             const double originsLongitudeInRad,
                                             const double pointsLatitudeInRad,
                                             const double pointsLongitudeInRad,
                                             double& pointsX,
                                             double& pointsY);

    static void transformToTangentialPlane3d(const double originsLatitudeInRad,
                                             const double originsLongitudeInRad,
                                             const double originsAltitudeInMeter,
                                             const double pointsLatitudeInRad,
                                             const double pointsLongitudeInRad,
                                             const double pointsAltitudeInMeter,
                                             double& pointsX,
                                             double& pointsY,
                                             double& pointsZ);

public:
    static void transformToTangentialPlane2d(const EcefPoint& originRefPointsEcef,
                                             const double pointsLatitudeInRad,
                                             const double pointsLongitudeInRad,
                                             double& pointsX,
                                             double& pointsY);

    static void transformToTangentialPlaneF3d(const EcefPoint& originRefPointsEcef,
                                              const double pointsLatitudeInRad,
                                              const double pointsLongitudeInRad,
                                              const double pointsAltitudeInMeter,
                                              double& pointsX,
                                              double& pointsY,
                                              double& pointsZ);

public:
    static void transformFromTangentialPlane2d(const double originsLatitudeInRad,
                                               const double originsLongitudeInRad,
                                               const double pointsX,
                                               const double pointsY,
                                               double& pointsLatInRad,
                                               double& pointsLonInRad);

public:
    static void transformFromTangentialPlane3d(const double originsLatitudeInRad,
                                               const double originsLongitudeInRad,
                                               const double originsAltitude,
                                               const double pointsX,
                                               const double pointsY,
                                               const double pointsZ,
                                               double& pointsLatInRad,
                                               double& pointsLonInRad,
                                               double& pointsAltInMeter);

public:
    PositionWgs84();
    virtual ~PositionWgs84();

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // setter
        //========================================
        //! \name Setter methods
        //! Use these methods to change the object'S data fields.
        //----------------------------------------
    //@{

    //========================================
    //! \brief Sets the time when the position measurement was taken.
    //!
    //! The time should be as close to the real measurement time as possible.
    //!
    //! \param[in] usSinceStartup  Elapsed microseconds since device startup.
    //----------------------------------------
    void setUsSinceStartup(const uint32_t usSinceStartup) { m_usSinceStartup = usSinceStartup; }

    //========================================
    //! \brief Sets the time when the position measurement was taken.
    //!
    //! The time should be as close to the real measurement time as possible.
    //!
    //! \param[in] newTimeStamp  Timestamp of this message.
    //----------------------------------------
    void setTimestamp(const NtpTime newTimeStamp) { m_timestamp = newTimeStamp; }

    //========================================
    //! \brief Sets the device id of the source device.
    //!
    //! \param[in] deviceId  New device id.
    //----------------------------------------
    void setDeviceId(const uint8_t deviceId) { m_deviceId = deviceId; }

    //========================================
    //! \brief Sets the latitude value of the WGS-84 position
    //! \param[in] val  Latitude value in [rad]. Must be in the interval
    //!                 [-Pi/2, Pi/2] radians which corresponds to [-90,90] degree.
    //----------------------------------------
    void setLatitudeInRad(const double val) { m_latitude = val; }

    //========================================
    //! \brief Sets the latitude value of the WGS-84 position.
    //! \param[in] valDeg  Latitude value in [deg]. Must be in the interval
    //!                 [-90,90] degree.
    //----------------------------------------
    void setLatitudeInDeg(const double valDeg)
    {
        this->setLatitudeInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    //========================================
    //! \brief Sets the latitude value of the WGS-84 position.
    //! \param[in] Dm  Latitude value [0...9000] where format is DDmm.mmmmm.
    //!                D stands for value in degree and m for value in decimal minutes.
    //! \param[in] H   Hemisphere. For Latitude it can be [N]orth or [S]outh.
    //----------------------------------------
    void setLatitudeInNMEA(const double Dm, const char H);

    //========================================
    //! \brief Sets the latitude value of the WGS-84 position.
    //! \param[in] Dm  Latitude value in DDmm.mmmmm, where D stands
    //!                for value in degree and m for value in decimal minutes (signed).
    //----------------------------------------
    void setLatitudeInNMEA(const double Dm) { this->setLatitudeInNMEA(Dm, 'N'); }

    //========================================
    //! \brief Sets the sigma value of the normal distribution describing
    //!        the confidence about the latitude measurement.
    //!
    //! Negative values mean that no accuracy knowledge is available.
    //!
    //! \param[in] val  Sigma value in [m].
    //!
    //! \attention Sigma value is stored in meter. If you use the value
    //!            in combination with the latitude value be aware that
    //!            the units differ!
    //----------------------------------------
    void setLatitudeSigmaInMeter(const double val) { m_latitudeSigma = val; }

    //========================================
    //! \brief Sets the longitude value of the WGS-84 position.
    //!
    //! \param[in] val  Longitude value in [rad]. Must be in the range
    //!                 [-Pi, Pi] radians which corresponds to [-180, 180]
    //!                  degree.
    //----------------------------------------
    void setLongitudeInRad(const double val) { m_longitude = val; }

    //========================================
    //! \brief Sets the longitude value of the WGS-84 position.
    //!
    //! \param[in] valDeg  Longitude value in [deg]. Must be in the range
    //!                 [-180, 180] degree.
    //----------------------------------------
    void setLongitudeInDeg(const double valDeg)
    {
        this->setLongitudeInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    //========================================
    //! \brief Sets the longitude value of the WGS-84 position.
    //!
    //! \param[in] Dm  Latitude value in DDmm.mmmmm, where D stands
    //!                for value in degree and m for value in decimal
    //!                minutes.
    //! \param[in] H   Hemisphere. For longitude it can be
    //!                [W]est or [E]ast.
    //----------------------------------------
    void setLongitudeInNMEA(const double Dm, const char H);

    //========================================
    //! \brief Sets the longitude value of the WGS-84 position.
    //!
    //! \param[in] Dm  Latitude value in DDmm.mmmmm, where D
    //!                stands for value in degree and m for
    //!                value in decimal minutes (signed).
    //----------------------------------------
    void setLongitudeInNMEA(const double Dm) { this->setLongitudeInNMEA(Dm, 'E'); }

    //========================================
    //! \brief Sets the sigma value of the normal distribution describing
    //!        the confidence about the longitude measurement.
    //!
    //! Negative values mean that no accuracy knowledge is available.
    //!
    //! \param[in] val  Sigma value in [m].
    //!
    //! \attention Sigma value is stored in meter. If you use the value
    //!            in combination with the longitude value be aware that
    //!            the units differ!
    //----------------------------------------
    void setLongitudeSigmaInMeter(const double val) { m_longitudeSigma = val; }

    //========================================
    //! \brief Sets altitude value.
    //!
    //! This function sets the altitude above the mean sea level (MSL) in [m].
    //!
    //! \param[in] val  Altitude value in [m].
    //----------------------------------------
    void setAltitudeInMeterMSL(const double val) { m_altitudeMSL = val; }

    //========================================
    //! \brief Sets the sigma value of the normal distribution describing
    //!        the confidence about the altitude measurement.
    //!
    //! Negative values mean that no accuracy knowledge is available.
    //!
    //! \param[in] val  Sigma value in [m].
    //----------------------------------------
    void setAltitudeSigmaInMeterMSL(const double val) { m_altitudeMSLSigma = val; }

    //========================================
    //! \brief Sets the course angle.
    //!
    //! \note The input value will be normalized to [0, twoPi] if required.
    //!
    //! \param[in] val  Course angle value in [rad].
    //----------------------------------------
    void setCourseAngleInRad(const double val)
    {
        if ((val < 0.0) || (val > twoPi))
        {
            m_courseAngle = microvision::common::sdk::normalizeRadians(val);
            if (m_courseAngle < 0.0)
            {
                m_courseAngle += twoPi;
            }
        }
        else
        {
            m_courseAngle = val;
        }
    }

    //========================================
    //! \brief Sets the course angle.
    //!
    //! \param[in] valDeg  Course angle value in [deg]. (0 = North, 90 = West)
    //----------------------------------------
    void setCourseAngleInDeg(const double valDeg)
    {
        this->setCourseAngleInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    //========================================
    //! \brief Sets the sigma of the normal distribution describing
    //!        the confidence about the course angle.
    //!
    //! Negative values mean that no accuracy knowledge is available.
    //!
    //! \param[in] val  Sigma value in [rad].
    //----------------------------------------
    void setCourseAngleSigmaInRad(const double val) { m_courseAngleSigma = val; }

    //========================================
    //! \brief Sets the sigma of the normal distribution describing
    //!        the confidence about the course angle.
    //!
    //! Negative values mean that no accuracy knowledge is available.
    //!
    //! \param[in] valDeg  Sigma value in [deg].
    //!
    //! \sa setCourseAngleInRad()
    //! \sa setCourseAngleInDeg()
    //----------------------------------------
    void setCourseAngleSigmaInDeg(const double valDeg)
    {
        this->setCourseAngleSigmaInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    //========================================
    //! \brief Set the yaw angle in radians. Yaw angle is
    //!        automatically normalized to [0..2//!PI]
    //! \param[in] val  Yaw angle in [rad].
    //----------------------------------------
    void setYawAngleInRad(const double val)
    {
        m_yawAngle = microvision::common::sdk::normalizeRadians(val);
        if (m_yawAngle <= 0.0)
        {
            m_yawAngle += twoPi;
        }
    }

    //========================================
    //! \brief Set the yaw angle in radians. Yaw angle is
    //!        automatically normalized to [0..2//!PI]
    //! \param[in] valDeg  Yaw angle in [deg].
    //----------------------------------------
    void setYawAngleInDeg(const double valDeg)
    {
        this->setYawAngleInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    void setYawAngleSigmaInRad(const double val) { m_yawAngleSigma = val; }
    void setYawAngleSigmaInDeg(const double valDeg)
    {
        this->setYawAngleSigmaInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    void setPitchAngleInRad(const double val) { m_pitchAngle = val; }
    void setPitchAngleInDeg(const double valDeg)
    {
        this->setPitchAngleInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    void setPitchAngleSigmaInRad(const double val) { m_pitchAngleSigma = val; }
    void setPitchAngleSigmaInDeg(const double valDeg)
    {
        this->setPitchAngleSigmaInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    void setRollAngleInRad(const double val) { m_rollAngle = val; }
    void setRollAngleInDeg(const double valDeg)
    {
        this->setRollAngleInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    void setRollAngleSigmaInRad(const double val) { m_rollAngleSigma = val; }
    void setRollAngleSigmaInDeg(const double valDeg)
    {
        this->setRollAngleSigmaInRad(unit::Convert<unit::angle::degree, unit::angle::radian, double>()(valDeg));
    }

    //========================================
    //! \brief Sets the source of the position measurement.
    //! \param[in] val Source of measurement.
    //----------------------------------------
    void setSource(const SourceType val) { m_source = val; }

    //@}

public:
    //========================================
    //! \name Getter methods
    //! Use these methods to retrieve position information.
    //----------------------------------------
    //@{

    //========================================
    //! \brief Get the microseconds since device startup
    //!        when the position measurement was taken.
    //!
    //! The time should be as close to the real measurement time as possible.
    //!
    //! \return Microseconds since startup.
    //----------------------------------------
    uint32_t getUsSinceStartup() const { return m_usSinceStartup; }

    //========================================
    //! \brief Get the timestamp for the moment
    //!        when the position measurement was taken.
    //!
    //! The time should be as close to the real measurement time as possible.
    //!
    //! \return Timestamp of this measurement.
    //----------------------------------------
    NtpTime getTimestamp() const { return m_timestamp; }

    //========================================
    //! \brief Get the device id of the device which has
    //!        taken this measurement
    //! \return Source device id.
    //----------------------------------------
    uint8_t getDeviceId() const { return m_deviceId; }

    //========================================
    //! \brief Returns the time when the position measurement was taken.
    //!
    //! The time should be as close to the real measurement time as possible.
    //!
    //! \return UTC time stamp of measurement recording time.
    //----------------------------------------
    boost::posix_time::ptime getTimestampPosix() const { return m_timestamp.toPtime(); }

    //========================================
    //! \brief Returns the latitude value of the WGS-84 position.
    //!
    //! \return Latitude value in [rad].
    //----------------------------------------
    double getLatitudeInRad() const { return m_latitude; }

    //========================================
    //! \brief  Returns the latitude value of the WGS-84 position.
    //! \return Latitude value in [deg].
    //----------------------------------------
    double getLatitudeInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_latitude);
    }

    //========================================
    //! \brief  Returns the latitude sigma value of the WGS-84 position.
    //! \return Latitude sigma value in [m].
    //----------------------------------------
    double getLatitudeSigmaInMeter() const { return m_latitudeSigma; }

    //========================================
    //! \brief Returns the longitude value of the WGS-84 position.
    //! \return Longitude value in [rad].
    //----------------------------------------
    double getLongitudeInRad() const { return m_longitude; }

    //========================================
    //! \brief Returns the longitude value of the WGS-84 position.
    //! \return Longitude value in [deg].
    //----------------------------------------
    double getLongitudeInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_longitude);
    }

    //========================================
    //! \brief  Returns the longitude sigma value of the WGS-84 position.
    //! \return Longitude sigma value in [m].
    //----------------------------------------
    double getLongitudeSigmaInMeter() const { return m_longitudeSigma; }

    //========================================
    //! \brief Returns the altitude in meter above mean sea level.
    //! \return Altitude in [m] above mean sea level (MSL).
    //----------------------------------------
    double getAltitudeInMeterMSL() const { return m_altitudeMSL; }

    //========================================
    //! \brief Returns the altitude sigma value.
    //! \return Altitude sigma in [m].
    //----------------------------------------
    double getAltitudeSigmaInMeterMSL() const { return m_altitudeMSLSigma; }

    //========================================
    //! \brief Returns the course angle in [rad].
    //!
    //! The course angle is the angle the vehicle is traveling to.
    //! If you drift, it's different to the yaw angle, which is the direction
    //! of the vehicle is heading/looking at.
    //!
    //! The angle in between is called slip angle.
    //!
    //! The angle is always counted positive in counter-clockwise direction, since
    //! in our system the z-axis is pointing upwards.
    //!
    //! Example: If the car is heading to the north, but driving backwards,
    //! the yaw angle is 0 degrees and the course angle 180 degrees.
    //!
    //! \return Course angle in [rad]. (0 = North, pi/2 = West)
    //----------------------------------------
    double getCourseAngleInRad() const { return m_courseAngle; }

    //========================================
    //! \brief Returns the course angle in [deg].
    //!
    //! The course angle is the angle the vehicle is travelling to.
    //! If you drift, it's different to the yaw angle, which is the direction
    //! of the vehicle is heading/looking at.
    //!
    //! The angle in between is called slip angle.
    //!
    //! The angle is always counted positive in counter-clockwise direction, since
    //! in our system the z-axis is pointing upwards.
    //!
    //! Example: If the car is heading to the north, but driving backwards,
    //! the yaw angle is 0 degrees and the course angle 180 degrees.
    //!
    //! \return Course angle in [deg]. (0 = North, 90 = West)
    //----------------------------------------
    double getCourseAngleInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_courseAngle);
    }

    //========================================
    //! \brief Returns the course angle sigma.
    //! \return Course angle sigma in [rad]
    //----------------------------------------
    double getCourseAngleSigmaInRad() const { return m_courseAngleSigma; }

    //========================================
    //! \brief Returns the course angle sigma.
    //! \return Course angle sigma in [deg]
    //----------------------------------------
    double getCourseAngleSigmaInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(getCourseAngleSigmaInRad());
    }

    //========================================
    //! \brief Returns the yaw angle in [rad].
    //!
    //! The yaw angle is the angle the vehicle is heading/looking at.
    //! If you drift, it's different to the course angle, which is the direction
    //! of travelling or the track angle.
    //!
    //! The angle in between is called slip angle.
    //!
    //! The angle is always counted positive in counter-clockwise direction, since
    //! in our system the z-axis is pointing upwards.
    //!
    //! Example: If the car is heading to the north, but driving backwards,
    //! the yaw angle is 0 degrees and the course angle 180 degrees.
    //!
    //! \return Yaw angle in [rad]. (0 = North, pi/2 = West)
    //----------------------------------------
    double getYawAngleInRad() const { return m_yawAngle; }

    //========================================
    //! \brief Returns the yaw angle.
    //!
    //! The yaw angle is the angle the vehicle is heading/looking at.
    //! If you drift, it's different to the course angle, which is the direction
    //! of travelling or the track angle.
    //!
    //! The angle in between is called slip angle.
    //!
    //! The angle is always counted positive in counter-clockwise direction, since
    //! in our system the z-axis is pointing upwards.
    //!
    //! Example: If the car is heading to the north, but driving backwards,
    //! the yaw angle is 0 degrees and the course angle 180 degrees.
    //!
    //! \return Yaw angle in [deg]. (0 = North, 90 = West)
    //----------------------------------------
    double getYawAngleInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_yawAngle);
    }

    //========================================
    //! \brief Returns the yaw sigma.
    //! \return Yaw sigma in [rad]
    //----------------------------------------
    double getYawAngleSigmaInRad() const { return m_yawAngleSigma; }

    //========================================
    //! \brief Returns the yaw sigma.
    //! \return Yaw sigma in [deg]
    //----------------------------------------
    double getYawAngleSigmaInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_yawAngleSigma);
    }

    double getPitchAngleInRad() const { return m_pitchAngle; }
    double getPitchAngleInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_pitchAngle);
    }

    double getPitchAngleSigmaInRad() const { return m_pitchAngleSigma; }
    double getPitchAngleSigmaInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_pitchAngleSigma);
    }

    double getRollAngleInRad() const { return m_rollAngle; }
    double getRollAngleInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_rollAngle);
    }

    double getRollAngleSigmaInRad() const { return m_rollAngleSigma; }
    double getRollAngleSigmaInDeg() const
    {
        return unit::Convert<unit::angle::radian, unit::angle::degree, double>()(m_rollAngleSigma);
    }

    //========================================
    //! \brief Returns the type of source that
    //!        identifies the type of the device
    //!        that created this object.
    //! \return Type of source that created this object.
    //----------------------------------------
    SourceType getSource() const { return m_source; }

    std::string getSourceString() const;
    //@}

public:
    void transformToTangentialPlane2d(const PositionWgs84& origin, double& pdX, double& pdY) const;

public:
    void transformToTangentialPlane2d(const double originLatitudeInRad,
                                      const double originLongitudeInRad,
                                      double& pdX,
                                      double& pdY) const;

public:
    void transformFromTangentialPlane2d(const double x, const double y, const PositionWgs84& origin);

public:
    //========================================
    //! \brief See transformToTangentialPlane, but here, the heading of the origin is
    //!         respected, thus the resulting coordinates are relative to the coordinate
    //!         system given by the origin including the orientation
    //!
    //! \param[in]  origin  The reference coordinate system
    //! \param[out] x       The distance in x-direction
    //! \param[out] y       The distance in y-direction
    //!
    //! \sa transformToTangentialPlane()
    //----------------------------------------
    void transformToTangentialPlaneWithHeading2d(const PositionWgs84& origin, double& x, double& y) const;

public:
    //========================================
    //! \brief see transformFromTangentialPlane, but here, the heading of the origin is
    //!         respected, thus the coordinates have to be given in this system
    //!
    //! \param[in] x       The distance in x-direction
    //! \param[in] y       The distance in y-direction
    //! \param[in] origin  The reference coordinate system
    //!
    //! \sa transformFromTangentialPlane()
    //----------------------------------------
    void transformFromTangentialPlaneWithHeading2d(const double x, const double y, const PositionWgs84& origin);

protected:
    // some constants for the ellipsoid called earth
    static const double a; //!< SemiMajorAxis
    static const double b; //!< SemiMinorAxis
    //static const double f; // !< ellipsoid flattening
    static const double ea2; //!< first eccentricity
    static const double eb2; //!< second eccentricity

protected:
    uint32_t m_usSinceStartup{0}; //!< Microseconds since device startup.
    NtpTime m_timestamp{0};

    uint8_t m_deviceId{0}; //!< id of device

    double m_latitude{NaN_double}; //!< [rad]
    double m_latitudeSigma{NaN_double}; //!< [m]
    double m_longitude{NaN_double}; //!< [rad]
    double m_longitudeSigma{NaN_double}; //!< [m]
    double m_altitudeMSL{NaN_double}; //!< [m]
    double m_altitudeMSLSigma{NaN_double}; //!< [m]

    double m_courseAngle{NaN_double}; //!< [rad]
    double m_courseAngleSigma{NaN_double}; //!< [rad]

    double m_yawAngle{NaN_double}; //!< [rad]
    double m_yawAngleSigma{NaN_double}; //!< [rad]
    double m_pitchAngle{NaN_double}; //!< [rad]
    double m_pitchAngleSigma{NaN_double}; //!< [rad]
    double m_rollAngle{NaN_double}; //!< [rad]
    double m_rollAngleSigma{NaN_double}; //!< [rad]

    SourceType m_source{SourceType::Unknown}; //!< Source of position information+ 2 Bytes (encoded as uint16_t)
}; // PositionWgs84

//==============================================================================

bool operator==(const PositionWgs84& lhs, const PositionWgs84& rhs);
bool operator!=(const PositionWgs84& lhs, const PositionWgs84& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

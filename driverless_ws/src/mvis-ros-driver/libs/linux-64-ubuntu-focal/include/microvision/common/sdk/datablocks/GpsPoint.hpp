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
//! \date Mar 14, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \class GpsPoint
//! \brief A single point with GPS coordinates (WGS84).
//! \date Mar 14, 2016
//------------------------------------------------------------------------------
class GpsPoint final
{
public:
    static std::streamsize getSerializedSize_static();

public:
    GpsPoint();
    GpsPoint(const PositionWgs84& wgs84Point);
    GpsPoint(const double& lonInDeg, const double& latInDeg, const float altitude);

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public:
    bool operator==(const GpsPoint& other) const;
    bool operator!=(const GpsPoint& other) const;

public: // getter
    double getLongitudeInDeg() const { return m_longitude; }
    double getLatitudeInDeg() const { return m_latitude; }

    double getLongitudeInRad() const
    {
        return unit::Convert<unit::angle::degree, unit::angle::radian, double>()(m_longitude);
    }
    double getLatitudeInRad() const
    {
        return unit::Convert<unit::angle::degree, unit::angle::radian, double>()(m_latitude);
    }

    float getAltitude() const { return m_altitude; }

public: // setter
    void setLongitudeInDeg(const double valueInDeg) { m_longitude = valueInDeg; }
    void setLatitudeInDeg(const double valueInDeg) { m_latitude = valueInDeg; }

    void setLongitudeInRad(const double value)
    {
        m_longitude = unit::Convert<unit::angle::radian, unit::angle::degree, double>()(value);
    }
    void setLatitudeInRad(const double value)
    {
        m_latitude = unit::Convert<unit::angle::radian, unit::angle::degree, double>()(value);
    }

    void setAltitude(const float value) { m_altitude = value; }

    //========================================
    //! \brief Get this gps point position in WGS84 format.
    //! \param[in] courseAngle Course angle needed for the WGS84 position.
    //! \return WGS84 Position of this gps point with given course angle.
    //----------------------------------------
    PositionWgs84 getAsPositionWGS84(const float courseAngle = .0f) const;

private:
    // WGS84 coordinates
    double m_latitude; // deg
    double m_longitude; // deg
    float m_altitude; // meter
}; // GpsPoint

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

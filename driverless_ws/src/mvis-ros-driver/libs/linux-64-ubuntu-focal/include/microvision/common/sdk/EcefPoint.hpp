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
//! \date Mar 17, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector3.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief "Earth centered earth fixed" point (aka ECR "earth centered rotational")
//! \date Mar 17, 2016
//!
//! Class to store the ECEF coordinates of a GPS point.
//! For performance reasons it also holds the sine and cosine
//! of the GPS point's latitude and longitude.
//------------------------------------------------------------------------------
class EcefPoint final
{
public: // constructor
    EcefPoint();
    EcefPoint(const double x, const double y, const double z, const double latRad, const double lonRad);

    EcefPoint(const Vector3<double>& xyz, const double latRad, const double lonRad);

    EcefPoint(const Vector3<double>& xyz,
              const double latSin,
              const double latCos,
              const double lonSin,
              const double lonCos);

    EcefPoint(const double x,
              const double y,
              const double z,
              const double latSin,
              const double latCos,
              const double lonSin,
              const double lonCos);

public: // getter
    const Vector3<double>& getXyz() const { return m_xyz; }
    double getX() const { return m_xyz.getX(); }
    double getY() const { return m_xyz.getY(); }
    double getZ() const { return m_xyz.getZ(); }
    double getLatSin() const { return m_latSin; }
    double getLatCos() const { return m_latCos; }
    double getLonSin() const { return m_lonSin; }
    double getLonCos() const { return m_lonCos; }

public: // setter
    void set(const double x, const double y, const double z, const double latRad, const double lonRad);

    void set(const Vector3<double>& xyz, const double latRad, const double lonRad);

    void
    set(const Vector3<double>& xyz, const double latSin, const double latCos, const double lonSin, const double lonCos);

    void set(const double x,
             const double y,
             const double z,
             const double latSin,
             const double latCos,
             const double lonSin,
             const double lonCos);

protected:
    Vector3<double> m_xyz;
    double m_latSin;
    double m_latCos;
    double m_lonSin;
    double m_lonCos;
}; // EcefPoint

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

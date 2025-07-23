//==============================================================================
//! \file
//!
//! \brief Reference plane to be used as a global reference of a point cloud.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2016
//------------------------------------------------------------------------------
//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/GpsPoint.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808.hpp>
#include <microvision/common/sdk/RotationMatrix3d.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ReferencePlane final
{
public:
    ReferencePlane()
      : m_gpsPoint(), m_yaw(0.F), m_pitch(0.F), m_roll(0.F), m_rotationMatrixIsValid(false), m_rotationMatrix()
    {}

    ReferencePlane(const GpsPoint& point, const float yaw = 0.F, const float pitch = 0.F, const float roll = 0.F)
      : m_gpsPoint(point), m_yaw(yaw), m_pitch(pitch), m_roll(roll), m_rotationMatrixIsValid(false), m_rotationMatrix()
    {}

    ReferencePlane(const VehicleState2808& vsb);

    ~ReferencePlane() = default;

public:
    bool operator==(const ReferencePlane& other) const;
    bool operator!=(const ReferencePlane& other) const;

public:
    std::streamsize getSerializedSize() const;
    bool deserialize(std::istream& is);
    bool serialize(std::ostream& os) const;

public:
    GpsPoint& gpsPoint() { return m_gpsPoint; }
    const GpsPoint& getGpsPoint() const { return m_gpsPoint; }
    void setGpsPoint(const GpsPoint& point) { m_gpsPoint = point; }

    float getYaw() const { return m_yaw; }
    void setYaw(const float yaw)
    {
        m_rotationMatrixIsValid = false;
        m_yaw                   = yaw;
    }

    float getPitch() const { return m_pitch; }
    void setPitch(const float pitch)
    {
        m_rotationMatrixIsValid = false;
        m_pitch                 = pitch;
    }

    float getRoll() const { return m_roll; }
    void setRoll(const float roll)
    {
        m_rotationMatrixIsValid = false;
        m_roll                  = roll;
    }

    const RotationMatrix3d<float>& getRotationMatrix() const
    {
        if (!m_rotationMatrixIsValid)
        {
            m_rotationMatrix.setFromVectorWithRotationOrderRollPitchYaw({m_roll, m_pitch, m_yaw});
        }
        return m_rotationMatrix;
    }

    //========================================
private:
    GpsPoint m_gpsPoint;

    float m_yaw;
    float m_pitch;
    float m_roll;

private: // rotation matrix cache
    bool m_rotationMatrixIsValid;
    mutable RotationMatrix3d<float> m_rotationMatrix;
}; // ReferencePlane

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

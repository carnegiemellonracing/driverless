//==============================================================================
//! \file
//!
//! \brief Point representation of PointCloud7500.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 16, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/pointcloud/PointBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class PointCloudPointIn7500 final : public PointBase
{
public:
    PointCloudPointIn7500() : PointBase() {}
    virtual ~PointCloudPointIn7500() = default;

public:
    static std::streamsize getSerializedSize_static();

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    GpsPoint& gpsPoint() { return m_gpsPoint; }
    const GpsPoint& getGpsPoint() const { return m_gpsPoint; }
    void setGpsPoint(const GpsPoint& point) { m_gpsPoint = point; }

private:
    GpsPoint m_gpsPoint;
}; // PointCloudPointIn7500

//==============================================================================

inline bool operator==(const PointCloudPointIn7500& lhs, const PointCloudPointIn7500& rhs)
{
    return lhs.getGpsPoint() == rhs.getGpsPoint();
}

inline bool operator!=(const PointCloudPointIn7500& lhs, const PointCloudPointIn7500& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

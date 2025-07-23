//==============================================================================
//! \file
//!
//! \brief Data package to store a cloud of points with respect to a reference plane system.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloudPointIn7510.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Point cloud plane.
//!
//! The point cloud datatype is used for holding a collection of 3-dimensional points.
//! The term plane is indicating that the data are stored with an offset to a reference system.
//! The differentiation is made because the reference system can also hold geo-coordinates and therefore
//! it is important, that this point cloud type does not store the geo-coordinates for each single point.
//!
//! General data type: \ref microvision::common::sdk::PointCloud
//------------------------------------------------------------------------------
class PointCloud7510 final : public SpecializedDataContainer, public PointCloudBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using PointVector = std::vector<PointCloudPointIn7510>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.pointcloud7510"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    PointCloud7510() : SpecializedDataContainer() {}
    virtual ~PointCloud7510() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    const PointVector& getPoints() const { return m_points; }
    void setPoints(const PointVector& points) { m_points = points; }

private:
    PointVector m_points;
}; // PointCloud7510

//==============================================================================

inline bool operator==(const PointCloud7510& lhs, const PointCloud7510& rhs)
{
    return (static_cast<const PointCloudBase&>(lhs) == static_cast<const PointCloudBase&>(rhs))
           && (lhs.getPoints() == rhs.getPoints());
}

//==============================================================================

inline bool operator!=(const PointCloud7510& lhs, const PointCloud7510& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

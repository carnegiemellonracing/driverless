//==============================================================================
//! \file
//!
//! \brief Data package to store a cloud of points with respect to a reference
//!        plane system and a tile offset.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 08, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudPoint.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Point cloud plane.
//!
//! The point cloud tile datatype is used for holding a collection of 3-dimensional points with a tile offset.
//! The term plane is indicating that the data are stored with an offset to a reference system.
//! The differentiation is made because the reference system can also hold geo-coordinates and therefore
//! it is important, that this point cloud type does not store the geo-coordinates for each single point.
//! The term tile is indicating that the data is stored with an offset to a reference system.
//! The differentiation is made to represent point cloud maps of a large range, which cannot
//! be hold within a single pointcloud using float points without suffering resolution errors.
//! Therefore, this datatype holds a tile of the full map, with a given tile offset to the maps origin.
//!
//! Special data types:
//! \ref microvision::common::sdk::PointCloud7500
//! \ref microvision::common::sdk::PointCloud7510
//! \ref microvision::common::sdk::PointCloud7511
//------------------------------------------------------------------------------
class PointCloud final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const PointCloud& lhs, const PointCloud& rhs);

public:
    using PointVector = PointCloud7511::PointVector;

public:
    //========================================
    //! \brief Unique (string) identifier of this class.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.pointcloud"};

    //========================================
    //! \brief Get the hash for this container type (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Default Constructor.
    //!
    //! Creates an empty point cloud.
    //----------------------------------------
    PointCloud() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~PointCloud() override = default;

public:
    //========================================
    //! \brief Get the hash for this container type.
    //!
    //! \return The hash value for this container type.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get points of the point cloud. Points are given
    //!        within local coordinate frame with respect to the tile offset.
    //!        Add tileOffset to points to calculate the global point coordinates.
    //! \return The point vector.
    //----------------------------------------
    const PointVector& getPoints() const { return m_delegate.getPoints(); }

    //========================================
    //! \brief Get a reference to the points of the point cloud. Points are given
    //!        within local coordinate frame with respect to the tile offset.
    //!        Add tileOffset to points to calculate the global point coordinates.
    //! \return The point vector.
    //----------------------------------------
    PointVector& points() { return m_delegate.points(); }

    //========================================
    //! \brief Set a  the points of the point cloud. Points are given
    //!        within local coordinate frame with respect to the tile offset
    //! \param [in] points  The point vector to be set.
    //----------------------------------------
    void setPoints(const PointVector& points) { m_delegate.setPoints(points); }

    //========================================
    //! \brief Get the tile coordinate offset with respect to the reference system.
    //!        This is an offset to be added to the points.
    //! \return The tile coordinate offset.
    //----------------------------------------
    const Vector3<double>& getTileOffset() const { return m_delegate.getTileOffset(); }

    //========================================
    //! \brief Get a reference to the tile coordinate offset
    //!        This is an offset to be added to the points.
    //! \return The tile coordinate offset.
    //----------------------------------------
    Vector3<double>& tileOffset() { return m_delegate.tileOffset(); }

    //========================================
    //! \brief Set the tile coordinate offset.
    //! \param [in] offset  The tile coordinate offset.
    //----------------------------------------
    void setTileOffset(const Vector3<double>& offset) { m_delegate.setTileOffset(offset); }

    //========================================
    //! \brief Get the scanner type, which has been used to create the point cloud.
    //! \return Scanner type that was used.
    //----------------------------------------
    ScannerType getScannerType() const { return m_delegate.getScannerType(); }

    //========================================
    //! \brief Set the scanner type, which has been used to create the point cloud.
    //! \param [in] type  Scanner type.
    //----------------------------------------
    void setScannerType(const ScannerType& type) { m_delegate.setScannerType(type); }

    //========================================
    //! \brief Get the kind of the point cloud points. This represents the classification
    //!        of the points, e.g raw scanpoints, lane marking points, etc.
    //! \return The kind of the points.
    //----------------------------------------
    PointKind getKind() const { return m_delegate.getKind(); }

    //========================================
    //! \brief Set the kind of the point cloud points. This represents the classification
    //!        of the pointd, e.g raw scanpoints, lane marking points, etc
    //! \param [in] kind  The kind of the points.
    //----------------------------------------
    void setKind(const PointKind kind) { m_delegate.setKind(kind); }

    //========================================
    //! \brief Get the type of the point cloud points. This indicates which fields are available
    //!        within the points: EPW, Flags, Color.
    //! \return The type of the points.
    //----------------------------------------
    PointType getType() const { return m_delegate.getType(); }

    //========================================
    //! \brief Set the type of the point cloud points. This indicates which fields are available
    //!        within the points: EPW, Flags, Color.
    //! \param [in] type  The type of the points.
    //----------------------------------------
    void setType(const PointType type) { m_delegate.setType(type); }

    //========================================
    //! \brief Checks if the point cloud points contain EPW information.
    //! \return true, if and only if EPW information is available within points.
    //----------------------------------------
    bool hasEpw() const { return m_delegate.hasEpw(); }

    //========================================
    //! \brief Checks if the point cloud points contain classification flags.
    //! \return true, if and only if flags is available within points.
    //----------------------------------------
    bool hasFlags() const { return m_delegate.hasFlags(); }

    //========================================
    //! \brief Checks if the point cloud points contain color information.
    //! \return true, if and only if color information is available within points.
    //----------------------------------------
    bool hasColor() const { return m_delegate.hasColor(); }

    //========================================
    //! \brief Get a reference to the global reference plane of the point cloud.
    //! \return A reference to the reference plane of the point cloud.
    //----------------------------------------
    ReferencePlane& referencePlane() { return m_delegate.referencePlane(); }

    //========================================
    //! \brief Get the global reference plane of the point cloud.
    //! \return The reference plane of the point cloud.
    //----------------------------------------
    const ReferencePlane& getReferencePlane() const { return m_delegate.getReferencePlane(); }

    //========================================
    //! \brief Set the global reference plane of the point cloud.
    //! \param [in] plane  The reference plane of the point cloud.
    //----------------------------------------
    void setReferencePlane(const ReferencePlane& plane) { m_delegate.setReferencePlane(plane); }

protected:
    PointCloud7511 m_delegate;
}; // PointCloudTile

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const PointCloud& lhs, const PointCloud& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const PointCloud& lhs, const PointCloud& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

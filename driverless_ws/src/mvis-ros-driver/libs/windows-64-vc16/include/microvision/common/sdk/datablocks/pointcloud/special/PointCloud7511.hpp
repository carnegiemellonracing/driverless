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
#include <microvision/common/sdk/datablocks/pointcloud/PointCloudBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloudPointIn7511.hpp>
#include <microvision/common/sdk/ScannerType.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Point cloud plane.
//!
//! The point cloud datatype is used for holding a collection of 3-dimensional points with a tile offset.
//! The term plane is indicating that the data are stored with an offset to a reference system.
//! The differentiation is made because the reference system can also hold geo-coordinates and therefore
//! it is important, that this point cloud type does not store the geo-coordinates for each single point.
//! The term tile is indicating that the data is stored with an offset to a reference system.
//! The differentiation is made to represent point cloud maps of a large range, which cannot
//! be hold within a single pointcloud using float points without suffering resolution errors.
//! Therefore, this datatype holds a tile of the full map, with a given tile offset to the maps origin.
//!
//! General data type: \ref microvision::common::sdk::PointCloud
//------------------------------------------------------------------------------
class PointCloud7511 final : public SpecializedDataContainer, public PointCloudBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using PointVector = std::vector<PointCloudPointIn7511>;

public:
    //========================================
    //! \brief Unique (string) identifier of this class.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.pointcloud7511"};

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
    PointCloud7511() = default;

    //========================================
    //! \brief Default Destructor.
    //----------------------------------------
    ~PointCloud7511() override = default;

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
    const PointVector& getPoints() const { return m_points; }

    //========================================
    //! \brief Get a reference to the points of the point cloud. Points are given
    //!        within local coordinate frame with respect to the tile offset.
    //!        Add tileOffset to points to calculate the global point coordinates.
    //! \return The point vector.
    //----------------------------------------
    PointVector& points() { return m_points; }

    //========================================
    //! \brief Set a  the points of the point cloud. Points are given
    //!        within local coordinate frame with respect to the tile offset
    //! \param [in] points  The point vector to be set.
    //----------------------------------------
    void setPoints(const PointVector& points) { m_points = points; }

    //========================================
    //! \brief Get the tile coordinate offset with respect to the reference system.
    //!        This is an offset to be added to the points.
    //! \return The tile coordinate offset.
    //----------------------------------------
    const Vector3<double>& getTileOffset() const { return m_tileOffset; }

    //========================================
    //! \brief Get a reference to the tile coordinate offset
    //!        This is an offset to be added to the points.
    //! \return The tile coordinate offset.
    //----------------------------------------
    Vector3<double>& tileOffset() { return m_tileOffset; }

    //========================================
    //! \brief Set the tile coordinate offset.
    //! \param [in] offset  The tile coordinate offset.
    //----------------------------------------
    void setTileOffset(const Vector3<double>& offset) { m_tileOffset = offset; }

    //========================================
    //! \brief Get the scanner type, which has been used to create the point cloud.
    //! \return Scanner type that was used.
    //----------------------------------------
    ScannerType getScannerType() const { return m_scannerType; }

    //========================================
    //! \brief Set the scanner type, which has been used to create the point cloud.
    //! \param [in] type  Scanner type.
    //----------------------------------------
    void setScannerType(const ScannerType& type) { m_scannerType = type; }

private:
    PointVector m_points{}; //!< Points of the tile, given in local coordinates with respect to the tile offset.
    Vector3<double> m_tileOffset{0.0, 0.0, 0.0}; //!< Coordinate offset of the tile [m].
    ScannerType m_scannerType{ScannerType::Invalid}; //!< Scanners that have been used to create this pointcloud.
}; // PointCloud7511

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const PointCloud7511& lhs, const PointCloud7511& rhs);

//==============================================================================
//! \brief Checks for inequality.
//!
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator!=(const PointCloud7511& lhs, const PointCloud7511& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

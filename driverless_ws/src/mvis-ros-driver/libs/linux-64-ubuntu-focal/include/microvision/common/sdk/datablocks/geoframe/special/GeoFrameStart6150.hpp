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
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/ReferencePlane.hpp>
#include <microvision/common/sdk/datablocks/geoFrameTypes.hpp>
#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief GeoFrameStart 0x6150 data type.
//!
//! An instance of this data type is starting a sequence of data types in the IDC
//! file, that belong to the same geo frame or tile.
//!
//! A geo-frame is defined to be started by a GeoFrameStart6150 and ends before the
//! next GeoFrameStart6150 instance or at the file end.
//!
//! All data types between belongs to the same geo-frame, describing a tile of the
//! tile map IDC file. E.g. a geo-frame can contain several PointCloud7510 data types
//! of different PointKind.
//!
//! As (size) layer id, the tile size is used.
//!
//! For more details see #GeoFrameIndex6140.
//------------------------------------------------------------------------------
class GeoFrameStart6150 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.geoframestart6150"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    //========================================
    //! \brief Calculate the index for a given point (\a x, \a y) and a given \a tileSize.
    //! \param[in] tileSize  The size of a tile in meter.
    //! \param[in] x         x coordinate of the point in meter.
    //! \param[in] y         y coordinate of the point in meter.
    //! \return The TileIndex of the given point and the given tile size.
    //----------------------------------------
    template<typename T>
    static TileIndex calcTileIndex(const TileSizeType tileSize, const T x, const T y)
    {
        return {calcTileIndexComponent(tileSize, x), calcTileIndexComponent(tileSize, y)};
    }

    //========================================
    //! \brief Calcualte the index for the given point at \a position and a given \a tileSize.
    //!
    //! Convenient function.
    //!
    //! \param[in] tileSize  The size of a tile in meter.
    //! \param[in] position  The given point for which the tile index shall be calculated.
    //!                      The coordinates are given in meter.
    //----------------------------------------
    template<typename T>
    static TileIndex calcTileIndex(const TileSizeType tileSize, const Vector2<T> position)
    {
        return calcTileIndex(tileSize, position.getX(), position.getY());
    }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    GeoFrameStart6150() = default;

    //========================================
    //! \brief Constructor
    //! \param[in] referencePlane  The reference plane used for this tiled map IDC file.
    //! \param[in] tileSize        The layer id and also tile size of this geo frame.
    //! \param[in] tileIndex       The index of the tile of this geo frame.
    //----------------------------------------
    GeoFrameStart6150(const ReferencePlane& referencePlane, const TileSizeType tileSize, const TileIndex tileIndex);

    //========================================
    //! \brief Destrutor.
    //----------------------------------------
    ~GeoFrameStart6150() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the reference plane use for this GeoFrameStart6150.
    //! \return The reference plane of this GeoFrameStart6150.
    //----------------------------------------
    const ReferencePlane& getReferencePlane() const { return m_referencePlane; }

    //========================================
    //! \brief Get the tile size of this geo-frame.
    //!
    //! The tile size is also serving as the (size) layer id.
    //! For more details see #GeoFrameIndex6140.
    //!
    //! \return The tile size in this geo-frame in meter.
    //----------------------------------------
    TileSizeType getTileSize() const { return m_tileSize; }

    //========================================
    //! \brief Get the tile index of this geo-frame.
    //! \return The tile index of this geo-frame.
    //----------------------------------------
    TileIndex getTileIndex() const { return m_tileIndex; }

public:
    //========================================
    //! \brief Set a new reference plane.
    //! \param[in] referencePlane  The new reference plane for
    //!                            this GeoFrameStart6150.
    //----------------------------------------
    void setReferencePlane(const ReferencePlane& referencePlane) { m_referencePlane = referencePlane; }

    //========================================
    //! \brief Set the tile size in meter.
    //!
    //! The tile size is also serving as the (size) layer id.
    //! For more details see #GeoFrameIndex6140.
    //!
    //! \param[in] tileSize  The new tile size in meter.
    //----------------------------------------
    void setTileSize(const TileSizeType tileSize) { m_tileSize = tileSize; }

    //========================================
    //! \brief Set the tile index.
    //! \param[in] tileIndex  The new tile index.
    //----------------------------------------
    void setTileIndex(const TileIndex tileIndex) { m_tileIndex = tileIndex; }

    //========================================
    //! \brief Set the tile index. The index is calculated from
    //!        a point in the reference plane.
    //!
    //! The index to be used will be calculated from the \a x, \a y coordinates of the point.
    //!
    //! \param[in] x  The x coordinate a point in meter.
    //! \param[in] y  The y coordinate a point in meter.
    //----------------------------------------
    template<typename T>
    void setTileIndex(const T x, const T y)
    {
        m_tileIndex = calcTileIndex(m_tileSize, x, y);
    }

private:
    //========================================
    //! \brief Calculate the index for one coordinate component.
    //! \param[in] tileSize  The tile size.
    //! \param[in] pos       The coordinate in one component.
    //! \return The index for one coordinate component.
    //----------------------------------------
    static int32_t calcTileIndexComponent(const TileSizeType tileSize, const float pos);

    //========================================
    //! \brief Calculate the index for one coordinate component.
    //! \param[in] tileSize  The tile size.
    //! \param[in] pos       The coordinate in one component.
    //! \return The index for one coordinate component.
    //----------------------------------------
    static int32_t calcTileIndexComponent(const TileSizeType tileSize, const double pos);

private:
    //========================================
    //! \brief The ReferencePlane used in the tile map IDC file.
    //----------------------------------------
    ReferencePlane m_referencePlane; // 32 bytes

    //========================================
    //! \brief The tile size in meter.
    //!
    //! The tile size is also serving as the (size) layer id.
    //! For more details see #GeoFrameIndex6140
    //----------------------------------------
    TileSizeType m_tileSize{0}; // 4 bytes

    //========================================
    //! \brief The index of the tile, in the following geo frame in the file.
    //----------------------------------------
    TileIndex m_tileIndex{0, 0}; // 8 bytes
}; // GeoFrameStart6150

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const GeoFrameStart6150& lhs, const GeoFrameStart6150& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(GeoFrameStart6150& lhs, const GeoFrameStart6150& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

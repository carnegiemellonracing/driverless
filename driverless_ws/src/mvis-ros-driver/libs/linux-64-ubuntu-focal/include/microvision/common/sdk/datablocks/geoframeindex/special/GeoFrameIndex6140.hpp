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
#include <microvision/common/sdk/misc/Optional.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief GeoFrameIndex 0x6140 data type.
//!
//! It holds the information about the tiles and their location inside an IDC file.
//!
//! For a given tile size, the tile for a point in the plane is well defined
//! and is represented by the tile index.
//!
//! Both together tile size and tile index are a unique identifier of a tile.
//!
//! The idea of having multiple tile size is, that smaller tiles can contain
//! more details than larger tiles. So when zooming out in a visualization
//! one can switch to larger tiles with less details but more tiles can be displayed.
//! Hence the tile size is serving as a level of refinement or (refinement) layer id.
//!
//! Content based layers are represented by the content of a tile (geo frame) in the
//! IDC file. There multiple PointCloud7510 can be stored each with different PointKind.
//------------------------------------------------------------------------------
class GeoFrameIndex6140 final : public microvision::common::sdk::SpecializedDataContainer
{
    template<class ContainerType, microvision::common::sdk::DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, microvision::common::sdk::DataTypeId::DataType id>
    friend class Exporter;

public:
    //========================================
    //! \brief Data type used to store a file position.
    //----------------------------------------
    using FilePosition = uint64_t;

    //========================================
    //! \brief For a given layer, i.e. tile size,
    //!        the map of the TileIndex to its file position.
    //----------------------------------------
    using LayerIndex = std::unordered_map<TileIndex, FilePosition>;

    //========================================
    //! \brief A map from the (size) layer id, the tile size, to the LayerIndex.
    //----------------------------------------
    using AllLayerIndex = std::unordered_map<TileSizeType, LayerIndex>;

public:
    //========================================
    //! \brief Container type as string.
    //----------------------------------------
    constexpr static const char* const containerType{"sdk.specialcontainer.geoframeindex6140"};

    //========================================
    //! \brief Hash value of this container (static version).
    //----------------------------------------
    static constexpr uint64_t getClassIdHashStatic() { return microvision::common::sdk::hash(containerType); }

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    GeoFrameIndex6140() = default;

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~GeoFrameIndex6140() override = default;

public:
    //========================================
    //! \brief Hash value of this container.
    //----------------------------------------
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    //========================================
    //! \brief Get the reference plane use for this GeoFrameIndex6140.
    //! \return The reference plane of this GeoFrameIndex6140.
    //----------------------------------------
    const ReferencePlane& getReferencePlane() const { return m_referencePlane; }

    //========================================
    //! \brief Set a new reference plane.
    //! \param[in] referencePlane  The new reference plane for
    //!                            this GeoFrameIndex6140.
    //----------------------------------------
    void setReferencePlane(const ReferencePlane& referencePlane) { m_referencePlane = referencePlane; }

public:
    //========================================
    //! \brief Get the index of all tile layers.
    //!
    //! A layer is identified by its (size) layer id, the tile size.
    //!
    //! \return The index of all tile layers.
    //----------------------------------------
    const AllLayerIndex& getAllLayerIndex() const { return m_tileLayers; }

    //========================================
    //! \brief Get the LayerIndex for the given \a tileSize (also size layer id).
    //! \param[in]  tileSize    The tile size (also size layer id) in meter.
    //! \return The layerIndex if the layer exists, \c nullptr otherwise.
    //----------------------------------------
    const LayerIndex* getLayerIndex(const TileSizeType tileSize) const;

    //========================================
    //! \brief Get the file position of the requested tile.
    //! \param[in]  tileSize  The tile size (also size layer id) of the requested tile in meter.
    //! \param[in]  index     The index of the requested tile.
    //! \return An optional containing the requested file position if the tile
    //!         exists. It is unset if not.
    //----------------------------------------
    Optional<FilePosition> getTileFilePosition(const TileSizeType tileSize, const TileIndex index) const;

    //========================================
    //! \brief Checks whether the given \a tileSize exists.
    //! \param[in] tileSize  The tile size (also size layer id) to be checked
    //!                      given in meter.
    //! \return \c True if the layer exists, \c false otherwise.
    //----------------------------------------
    bool hasLayer(const TileSizeType tileSize) const;

    //========================================
    //! \brief Checks whether the tile given by \a tileSize and \a index exists.
    //! \param[in] tileSize  The tile size (also size layer id) to be checked
    //!                      given in meter.
    //! \param[in] index     The index of the tile to be checked.
    //! \return \c True if the tile exists, \c false otherwise.
    //----------------------------------------
    bool hasTile(const TileSizeType tileSize, const TileIndex index) const;

public:
    //========================================
    //! \brief Add an index entry for a tile.
    //! \param[in] tileSize  The tile size  (also size layer id) in meter.
    //! \param[in] index     The index of the tile.
    //! \param[in] position  The file position of this tile.
    //!                      I.e. the file position of the GeoFrameStart6150
    //!                      data type for the tile identified by its tile size
    //!                      and its index.
    //----------------------------------------
    void addTile(const TileSizeType tileSize, const TileIndex index, const FilePosition position);

    //========================================
    //! \brief Add the index entries for several tiles with a given tile size.
    //! \param[in] tileSize  The tile size  (also size layer id) given in meter.
    //! \param[in] tiles     The index and file position of the tiles
    //!                      for which the index shall be added.
    //----------------------------------------
    void addTiles(const TileSizeType tileSize, const std::vector<std::pair<TileIndex, FilePosition>>& tiles);

    //========================================
    //! \brief Remove a tile for the index.
    //! \param[in] tileSize  The tile size (also size layer id) of the tile to be removed.
    //!                      The tile size is given in meter.
    //! \param[in] index     The index of the tile to be removed.
    //----------------------------------------
    void removeTile(const TileSizeType tileSize, const TileIndex index);

    //========================================
    //! \brief Remove all tiles with the given \a tileSize (also size layer id)
    //!        from the index.
    //! \param[in] tileSize  The tile size  (also size layer id) to be removed
    //!                      given in meter.
    //----------------------------------------
    void removeLayer(const TileSizeType tileSize);

    //========================================
    //! \brief Remove all layers and tiles from the index.
    //----------------------------------------
    void clear();

private:
    // GPS Point                  // 20
    // uint16_t nbOfLayers        // 2
    // L   TileSizeType(uint32_t) tileSize
    // L   uint32_t nbOfTiles
    //     T   int32/int32
    //     T   uint64

    //========================================
    //! \brief The ReferencePlane used in the tile map IDC file.
    //----------------------------------------
    ReferencePlane m_referencePlane;

    //========================================
    //! \brief A map to store the LayerIndex instances for each tile sizes.
    //!
    //! The tile size is serving as a (size) layer id.
    //----------------------------------------
    AllLayerIndex m_tileLayers;
}; // GeoFrameIndex6140

//==============================================================================

//==============================================================================
//! \brief Checks for equality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are identical, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const GeoFrameIndex6140& lhs, const GeoFrameIndex6140& rhs);

//==============================================================================
//! \brief Checks for inequality.
//! \param[in] lhs  The object, that shall be compared.
//! \param[in] rhs  The object, that first object shall be compared to.
//! \return \c True, if the objects are not identical, \c false otherwise.
//------------------------------------------------------------------------------
inline bool operator!=(const GeoFrameIndex6140& lhs, const GeoFrameIndex6140& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
//! \file
//!
//! \brief Data block to store zone definition information.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 29, 2025
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/zoneoccupationlist/special/RigidTransformationInA000.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class representing a zone definition with vertices and properties.
//------------------------------------------------------------------------------
class ZoneDefinitionInA000 final
{
public:
    //========================================
    //! \brief Maximum number of characters in zone name without trailing null-byte.
    //----------------------------------------
    static constexpr uint32_t maxLengthOfZoneName{64};

    //========================================
    //! \brief Maximum number of vertices allowed in a zone.
    //----------------------------------------
    static constexpr uint32_t maxNumberOfVertices{64};

    //========================================
    //! \brief Type definition for vertex coordinate vectors.
    //----------------------------------------
    using VertexCoordinateVector = std::vector<int32_t>;

    //========================================
    //! \brief Flag indicating that occlusion is treated as occupation.
    //----------------------------------------
    static constexpr uint32_t flagOcclusionIsOccupation = 0x00000001U;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ZoneDefinitionInA000() = default;

public:
    //========================================
    //! \brief Get zone ID.
    //! \return ID of the zone.
    //----------------------------------------
    int32_t getId() const;

    //========================================
    //! \brief Get zone name.
    //! \return Name of the zone.
    //----------------------------------------
    const std::string& getZoneName() const;

    //========================================
    //! \brief Get zone pose transformation.
    //! \return Rigid transformation representing the zone pose.
    //----------------------------------------
    const RigidTransformationInA000& getPose() const;

    //========================================
    //! \brief Get X coordinates of zone vertices.
    //! \return Vector of X coordinates in millimeters.
    //----------------------------------------
    const VertexCoordinateVector& getVerticesXInMm() const;

    //========================================
    //! \brief Get Y coordinates of zone vertices.
    //! \return Vector of Y coordinates in millimeters.
    //----------------------------------------
    const VertexCoordinateVector& getVerticesYInMm() const;

    //========================================
    //! \brief Get minimum Z extrusion value.
    //! \return Minimum Z coordinate of the extrusion in millimeters.
    //----------------------------------------
    int32_t getExtrusionMinZInMm() const;

    //========================================
    //! \brief Get maximum Z extrusion value.
    //! \return Maximum Z coordinate of the extrusion in millimeters.
    //----------------------------------------
    int32_t getExtrusionMaxZInMm() const;

    //========================================
    //! \brief Get maximum radial distance.
    //! \return Maximum radial distance from the sensor origin in millimeters.
    //----------------------------------------
    uint32_t getMaxRadialDistanceInMm() const;

    //========================================
    //! \brief Get multi-sampling value.
    //! \return Value for multi-sampling.
    //----------------------------------------
    uint32_t getMultiSampling() const;

    //========================================
    //! \brief Get zone flags.
    //! \return Bit-flags signaling further properties of the zone.
    //----------------------------------------
    uint32_t getFlags() const;

    //========================================
    //! \brief Get reserved value.
    //! \return Reserved value for future usage.
    //----------------------------------------
    uint32_t getReserved() const;

public:
    //========================================
    //! \brief Set zone ID.
    //! \param[in] id ID of the zone.
    //----------------------------------------
    void setId(const int32_t id);

    //========================================
    //! \brief Set zone name.
    //! \param[in] zoneName  Name of the zone.
    //! \return Either \c true if the length of \a zoneName is smaller than or equal to #maxLengthOfZoneName.
    //!         \c true otherwise.
    //----------------------------------------
    bool setZoneName(const std::string& zoneName);

    //========================================
    //! \brief Set zone pose transformation.
    //! \param[in] pose  Rigid transformation representing the zone pose.
    //----------------------------------------
    void setPose(const RigidTransformationInA000& pose);

    //========================================
    //! \brief Set zone vertices by copying coordinate vectors.
    //! \param[in] verticesXInMm  Vector of X coordinates in millimeters.
    //! \param[in] verticesYInMm  Vector of Y coordinates in millimeters.
    //! \return Either \c true if the number of x and y positions are identical and
    //!          less than or equal to #maxNumberOfVertices. \c true otherwise.
    //----------------------------------------
    bool setVertices(const VertexCoordinateVector& verticesXInMm, const std::vector<int32_t>& verticesYInMm);

    //========================================
    //! \brief Set zone vertices by moving coordinate vectors.
    //! \param[in] verticesXInMm  Vector of X coordinates in millimeters.
    //! \param[in] verticesYInMm  Vector of Y coordinates in millimeters.
    //! \return Either \c true if the number of x and y positions are identical and
    //!          less than or equal to #maxNumberOfVertices. \c true otherwise.
    //----------------------------------------
    bool setVertices(VertexCoordinateVector&& verticesXInMm, std::vector<int32_t>&& verticesYInMm);

    //========================================
    //! \brief Append a vertex to the zone.
    //! \param[in] xInMm  X coordinate in millimeters.
    //! \param[in] yInMm  Y coordinate in millimeters.
    //! \return Either \c true if the number of vertices on entry is smaller than
    //!          #maxNumberOfVertices. \c true otherwise.
    //----------------------------------------
    bool appendVertex(const int32_t xInMm, const int32_t yInMm);

    //========================================
    //! \brief Update an existing vertex.
    //! \param[in] indexOfVertex  Index of the vertex to update.
    //! \param[in] xInMm          New X coordinate in millimeters.
    //! \param[in] yInMm          New Y coordinate in millimeters.
    //! \return Either \c true if \a indexOfVertex is a valid index into the vector of vertices.
    //!         \c false otherwise.
    //----------------------------------------
    bool updateVertex(const uint32_t indexOfVertex, const int32_t xInMm, const int32_t yInMm);

    //========================================
    //! \brief Set minimum Z extrusion.
    //! \param[in] extrusionMinZInMm  Minimum Z coordinate of the extrusion in millimeters.
    //----------------------------------------
    void setExtrusionMinZInMm(const int32_t extrusionMinZInMm);

    //========================================
    //! \brief Set maximum Z extrusion.
    //! \param[in] extrusionMaxZInMm  Maximum Z coordinate of the extrusion in millimeters.
    //----------------------------------------
    void setExtrusionMaxZInMm(const int32_t extrusionMaxZInMm);

    //========================================
    //! \brief Set maximum radial distance.
    //! \param[in] maxRadialDistance  Maximum radial distance from the sensor origin in millimeters.
    //----------------------------------------
    void setMaxRadialDistanceInMm(const uint32_t maxRadialDistanceInMm);

    //========================================
    //! \brief Set multi-sampling value.
    //! \param[in] multiSampling Value for multi-sampling.
    //----------------------------------------
    void setMultiSampling(const uint32_t multiSampling);

    //========================================
    //! \brief Set zone flags.
    //! \param[in] flags Bit-flags signaling further properties of the zone.
    //----------------------------------------
    void setFlags(const uint32_t flags);

private:
    int32_t m_id{0}; //!< ID of the zone.
    std::string m_zoneName; //!< Name of the zone.

    RigidTransformationInA000 m_pose; //!< Rigid transformation representing the zone pose.
    VertexCoordinateVector m_verticesXInMm; //!< The x-coordinates of the vertices in millimeter.
    VertexCoordinateVector m_verticesYInMm; //!< The y-coordinates of the vertices in millimeter.

    int32_t m_extrusionMinZInMm{0}; //!< Minimum z-coordinate of the extrusion in millimeter.
    int32_t m_extrusionMaxZInMm{0}; //!< Maximum z-coordinate of the extrusion in millimeter.

    uint32_t m_maxRadialDistanceInMm{0}; //!< Maximum radial distance from the sensor origin in millimeter.
    uint32_t m_multiSampling{0}; //!< Value for multi-sampling.
    uint32_t m_flags{0}; //!< Bit-flags signaling further properties of the zone, nulled.

private:
    uint32_t m_reserved{0}; //!< Reserved for future usage, nulled. (Read only for now).
}; // ZoneDefinitionInA000

//==============================================================================
//! \brief Equality comparison operator for ZoneDefinitionInA000.
//! \param[in] lhs  Left-hand side operand.
//! \param[in] rhs  Right-hand side operand.
//! \return Either \c true if all zone properties are equal, \c false otherwise.
//------------------------------------------------------------------------------
bool operator==(const ZoneDefinitionInA000& lhs, const ZoneDefinitionInA000& rhs);

//==============================================================================
//! \brief Inequality comparison operator for ZoneDefinitionInA000.
//! \param[in] lhs  Left-hand side operand.
//! \param[in] rhs  Right-hand side operand.
//! \return Either \c true if any zone property differs, \c false if all are equal.
//------------------------------------------------------------------------------
inline bool operator!=(const ZoneDefinitionInA000& lhs, const ZoneDefinitionInA000& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

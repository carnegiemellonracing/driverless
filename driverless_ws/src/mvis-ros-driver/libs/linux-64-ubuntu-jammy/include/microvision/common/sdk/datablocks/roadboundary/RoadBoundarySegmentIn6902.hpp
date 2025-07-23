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
//! \date Okt 19, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryPointIn6902.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief A segment of n measurement points in a road boundary, representing e.g. a white dash on a road boundary.
//!
//! \sa microvision::common::sdk::RoadBoundaryList6902
//------------------------------------------------------------------------------
class RoadBoundarySegmentIn6902 final
{
public:
    //========================================
    //! \brief Vector of road boundaries in this list.
    //----------------------------------------
    using RoadBoundaryPoints = std::vector<RoadBoundaryPointIn6902>;

public:
    //========================================
    //! \brief Dash type of the segment.
    //----------------------------------------
    enum class RoadBoundaryDashType : uint8_t
    {
        NotSpecified         = 0x00U,
        ColoredLineSegment   = 0x01U,
        UncoloredLineSegment = 0x02U
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    RoadBoundarySegmentIn6902() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RoadBoundarySegmentIn6902() = default;

public: //getter + setter
    //========================================
    //! \brief Get a vector of road boundary points.
    //! \return The vector containing the road boundary points.
    //----------------------------------------
    const RoadBoundaryPoints& getRoadBoundaryPoints() const { return m_polyline; }

    //========================================
    //! \brief Get a vector of the road boundary points.
    //! \return The vector of road boundary points.
    //----------------------------------------
    RoadBoundaryPoints& getRoadBoundaryPoints() { return m_polyline; }

    //========================================
    //! \brief Get the dash type of the road boundary segment.
    //! \return The dash type of road boundary segment.
    //----------------------------------------
    const RoadBoundaryDashType& getRoadBoundaryDashType() const { return m_dashType; }

    //========================================
    //! \brief Sets a vector of new road boundary points.
    //! \param[in] points  The new points.
    //----------------------------------------
    void setRoadBoundaryPoints(const RoadBoundaryPoints& points) { m_polyline = points; }

    //========================================
    //! \brief Sets the dash type of the road boundary segment.
    //! \param[in] dashType  The new dash type.
    //----------------------------------------
    void setRoadBoundaryDashType(const RoadBoundaryDashType& dashType) { m_dashType = dashType; }

    //========================================
    //! \brief Adds a road boundary point at the end of the vector of road boundary points.
    //! \param[in] point  The added point.
    //----------------------------------------
    void addRoadBoundaryPoint(const RoadBoundaryPointIn6902& point) { m_polyline.push_back(point); }

private:
    RoadBoundaryPoints m_polyline; //!< A vector of the points in this road boundary segment.
    RoadBoundaryDashType m_dashType{
        RoadBoundaryDashType::NotSpecified}; //!< The dash type of the road boundary segment.

}; // RoadBoundarySegmentIn6902

//==============================================================================

bool operator==(const RoadBoundarySegmentIn6902& lhs, const RoadBoundarySegmentIn6902& rhs);

bool operator!=(const RoadBoundarySegmentIn6902& lhs, const RoadBoundarySegmentIn6902& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
//! \date Nov 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================
#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingPointIn6901.hpp>

#include <vector>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! \brief A segment of n measurement points in a lane marking, representing e.g. a white dash on a lane marking between lanes.
//!
//! \sa microvision::common::sdk::LaneMarkingList6901
//------------------------------------------------------------------------------
class LaneMarkingSegmentIn6901 final
{
public:
    //========================================
    //! \brief Vector of lane markings in this list..
    //----------------------------------------
    using LaneMarkingPoints = std::vector<LaneMarkingPointIn6901>;

public:
    //========================================
    //! \brief Dash type of the segment.
    //----------------------------------------
    enum class LaneMarkingDashType : uint8_t
    {
        NotSpecified         = 0x00U,
        ColoredLineSegment   = 0x01U,
        UncoloredLineSegment = 0x02U
    };

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    LaneMarkingSegmentIn6901() = default;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~LaneMarkingSegmentIn6901() = default;

public: //getter + setter
    //========================================
    //! \brief Get a vector of lane marking points.
    //! \return The vector containing the lane marking points.
    //----------------------------------------
    const LaneMarkingPoints& getLaneMarkingPoints() const { return m_polyline; }

    //========================================
    //! \brief Get a vector of the lane marking points.
    //! \return The vector of lane marking points.
    //----------------------------------------
    LaneMarkingPoints& getLaneMarkingPoints() { return m_polyline; }

    //========================================
    //! \brief Get the dash type of the lane marking segment.
    //! \return The dash type of lane marking segment.
    //----------------------------------------
    const LaneMarkingDashType& getLaneMarkingDashType() const { return m_dashType; }

    //========================================
    //! \brief Sets a vector of new lane marking points.
    //! \param[in] points  The new points.
    //----------------------------------------
    void setLaneMarkingPoints(const LaneMarkingPoints& points) { m_polyline = points; }

    //========================================
    //! \brief Sets the dash type of the lane marking segment.
    //! \param[in] dashType  The new dash type.
    //----------------------------------------
    void setLaneMarkingDashType(const LaneMarkingDashType& dashType) { m_dashType = dashType; }

    //========================================
    //! \brief Adds a lane marking point at the end of the vector of lane marking points.
    //! \param[in] point  The added point.
    //----------------------------------------
    void addLaneMarkingPoint(const LaneMarkingPointIn6901& point) { m_polyline.push_back(point); }

private:
    LaneMarkingPoints m_polyline; //!< A vector of the points in this lane marking segment.
    LaneMarkingDashType m_dashType{LaneMarkingDashType::NotSpecified}; //!< The dash type of the lane marking segment.

}; // LaneMarkingSegmentIn6901

//==============================================================================

bool operator==(const LaneMarkingSegmentIn6901& lhs, const LaneMarkingSegmentIn6901& rhs);

bool operator!=(const LaneMarkingSegmentIn6901& lhs, const LaneMarkingSegmentIn6901& rhs);

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

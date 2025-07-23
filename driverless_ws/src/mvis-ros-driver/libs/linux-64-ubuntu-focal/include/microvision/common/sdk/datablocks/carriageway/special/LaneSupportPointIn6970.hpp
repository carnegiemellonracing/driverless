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
//! \date Oct 9, 2014
//! \brief Support point for a LaneSegment
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>
#include <microvision/common/sdk/Vector2.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief This class represents a support point of a \ref LaneSegmentIn6970.
//!
//! A point holds information about gps position and heading and width
//! (more precisely the offsets to the left and right bounding line).
//------------------------------------------------------------------------------
class LaneSupportPointIn6970 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;

public:
    //! Default constructor, calls default constructors of members.
    LaneSupportPointIn6970() = default;

    //========================================
    //!\brief Constructor with initialization.
    //!\param[in] point            A GPS point with heading
    //!\param[in] lineOffsetLeft   The offset to the left bounding line
    //!\param[in] lineOffsetRight  The offset to the right bounding line
    //----------------------------------------
    LaneSupportPointIn6970(const PositionWgs84& point,
                           const Vector2<float>& lineOffsetLeft,
                           const Vector2<float>& lineOffsetRight)
      : m_latitudeInDeg{point.getLatitudeInDeg()},
        m_longitudeInDeg{point.getLongitudeInDeg()},
        m_courseAngleInDeg{point.getCourseAngleInDeg()},
        m_lineOffsetLeft{lineOffsetLeft},
        m_lineOffsetRight{lineOffsetRight}
    {}

public:
    //!\returns The GPS point and heading.
    PositionWgs84 getWgsPoint() const
    {
        PositionWgs84 p;
        p.setLongitudeInDeg(m_longitudeInDeg);
        p.setLatitudeInDeg(m_latitudeInDeg);
        p.setCourseAngleInDeg(m_courseAngleInDeg);
        return p;
    }

    double getLatitudeInDeg() const { return m_latitudeInDeg; }
    double getLongitudeInDeg() const { return m_longitudeInDeg; }
    double getCourseAngleInDeg() const { return m_courseAngleInDeg; }

    //!\returns The offset to the left marking start point.
    Vector2<float> getLeftOffset() const { return m_lineOffsetLeft; }

    //!\returns The offset to the right marking start point.
    Vector2<float> getRightOffset() const { return m_lineOffsetRight; }

private:
    double m_latitudeInDeg{0.0}; //!< Longitude position of the support point in deg.
    double m_longitudeInDeg{0.0}; //!< Latitude position of the support point in deg.
    double m_courseAngleInDeg{0.0}; //!< The course angle of the support point in deg (from north).
    Vector2<float> m_lineOffsetLeft{}; //!< The offset from the start point to the left lane marking.
    Vector2<float> m_lineOffsetRight{}; //!< The offset from the start point to the right lane marking.
}; // LaneSupportPointIn6970

//==============================================================================

inline bool operator==(const LaneSupportPointIn6970& lhs, const LaneSupportPointIn6970& rhs)
{
    return fuzzyDoubleEqualT<7>(lhs.getLatitudeInDeg(), rhs.getLatitudeInDeg()) //
           && fuzzyDoubleEqualT<7>(lhs.getLongitudeInDeg(), rhs.getLongitudeInDeg()) //
           && fuzzyDoubleEqualT<7>(lhs.getCourseAngleInDeg(), rhs.getCourseAngleInDeg()) //
           && (lhs.getLeftOffset() == rhs.getLeftOffset()) //
           && (lhs.getRightOffset() == rhs.getRightOffset());
}

//==============================================================================

inline bool operator!=(const LaneSupportPointIn6970& lhs, const LaneSupportPointIn6970& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

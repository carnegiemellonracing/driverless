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
//! \date Aug 16, 2018
//! \brief LaneHandler which implements functionality for the CarriageWay / Lane
//!       linked lists
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/LaneSegmentIn6970.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/LaneSegmentIn6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/LaneSegment.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleState.hpp>
#include <microvision/common/sdk/datablocks/PositionWgs84.hpp>
#include <microvision/common/sdk/TransformationMatrix2d.hpp>
#include <microvision/common/sdk/Line2d.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\class LaneHandler
//!\brief This class provides functionality for handling Lanes.
//!       It can be used to find the corresponding LaneSegment for a given gps point
//!       and calculates offsets to the center of the LaneSegment.
//!\date Oct 9, 2014
//------------------------------------------------------------------------------
template<typename TLaneSegment>
class LaneHandlerTemplate final
{
public:
    using LaneSegmentPtr = std::shared_ptr<TLaneSegment>;
    using CarriageWays   = std::vector<typename CarriageWayTemplate<TLaneSegment>::Ptr>;

    //==============================================================================
    //!\brief  Structure with information about offset to a laneSegment center.
    //------------------------------------------------------------------------------
    struct LanePosition
    {
        //========================================
        //! \brief Constructor.
        //----------------------------------------
        LanePosition() = default;

        //========================================
        //!\brief Pointer to the corresponding \ref LaneSegment.
        //----------------------------------------
        LaneSegmentPtr laneSegment;

        //========================================
        //!\brief The position and heading relative to the start of the segment.
        //----------------------------------------
        TransformationMatrix2d<float> transformationMatrix;

        //========================================
        //!\brief The lateral distance of to the center line of the \ref LaneSegment.
        //----------------------------------------
        float lateralDistance;

        //========================================
        //!\brief The longitudinal distance to the start of the \ref LaneSegment.
        //----------------------------------------
        float longitudinalDistance;

        //========================================
        //!\brief The WGS84 position;
        //----------------------------------------
        PositionWgs84 gpsPosition;
    }; // LanePosition

public:
    //========================================
    //!\brief Constructor
    //!\param[in] ways A list holding CarriageWays
    //----------------------------------------
    LaneHandlerTemplate(const CarriageWays& ways) : m_ways(ways)
    {
        for (auto& way : m_ways)
        {
            way->resolveConnections(way);
        }
    }

    //! default Constructor
    LaneHandlerTemplate() = default;

public: // getter and setter
    //========================================
    //!\brief Used to find a \ref LaneSegment by GPS coordinates of a given point and
    //!       returns information about distance to center of lane for the point
    //!       and distance from start of the segment along the lane to the point
    //!\param[in] point  The GPS point
    //----------------------------------------
    LanePosition getLanePosition(const PositionWgs84& point) const
    {
        LanePosition out{};
        out.gpsPosition = point;

        //========================================
        // find carriage ways with valid bounding boxes for point being inside
        CarriageWays validWays;
        for (const auto& way : m_ways)
        {
            BoundingRectangle mBox{way->getBoundingBox()};
            if (mBox.checkInside(point))
            {
                validWays.push_back(way);
            }
        }

        // return if no way was found
        if (validWays.empty())
        {
            return out;
        }

        //========================================
        // search valid CarriageWaySegments
        std::vector<typename CarriageWaySegmentTemplate<TLaneSegment>::Ptr> validCWSs;
        for (const auto& cw : validWays)
        {
            for (const auto& cwsme : cw->getCwSegmentsMap())
            {
                typename CarriageWaySegmentTemplate<TLaneSegment>::Ptr seg{cwsme.second};
                BoundingRectangle mBox{seg->getBoundingBox()};

                if (mBox.checkInside(point))
                {
                    validCWSs.push_back(seg);
                }
            }
        }

        // return if no way segment was found
        if (validCWSs.empty())
        {
            return out;
        }

        //========================================
        // search valid Lanes
        std::vector<typename LaneTemplate<TLaneSegment>::Ptr> validLanes;
        for (const auto& cws : validCWSs)
        {
            for (const auto& lme : cws->getLanesMap())
            {
                const typename LaneTemplate<TLaneSegment>::Ptr& lane{lme.second};
                BoundingRectangle mBox = lane->getBoundingBox();

                if (mBox.checkInside(point))
                {
                    validLanes.push_back(lane);
                }
            }
        }

        // return if no lane was found
        if (validLanes.empty())
        {
            return out;
        }

        //========================================
        // search valid Lane Segments
        std::vector<LaneSegmentPtr> validLaneSegments;
        for (const auto& lane : validLanes)
        {
            for (const auto& lsme : lane->getLaneSegmentsMap())
            {
                const LaneSegmentPtr& laneSeg{lsme.second};
                const BoundingRectangle mBox{laneSeg->getBoundingBox()};

                if (mBox.checkInside(point))
                {
                    validLaneSegments.push_back(laneSeg);
                }
            }
        }

        // return if no lane segment was found
        if (validLaneSegments.empty())
        {
            return out;
        }

        //========================================
        // now find the most suitable segment
        LanePosition bestInside{};
        LanePosition bestOutside{};
        float minOutsideDist{0.0F};
        bool bi{false};
        bool bo{false};

        for (const auto& curSeg : validLaneSegments)
        {
            bool inside{false};
            LanePosition curLP{calculateLanePosition(curSeg, point, inside)};

            // conditions intersection of projection is inside lane AND
            //            lateral distance is smaller than half of lane width in projection point AND
            //            the distance is closer than a previous found candidate
            if (inside
                && (std::abs(curLP.lateralDistance) <= (curLP.laneSegment->getWidth(curLP.longitudinalDistance) / 2)
                                                           + 0.001F) // add 1mm to avoid rounding errors
                && (!bi || std::abs(curLP.lateralDistance) < std::abs(bestInside.lateralDistance)))
            {
                bestInside = curLP;
                bi         = true;
            }
            else if (!inside)
            {
                float startDis{std::abs(curLP.longitudinalDistance)};
                float endDis{std::abs(curLP.longitudinalDistance - curSeg->getLength())};

                float useDis{startDis < endDis ? startDis : endDis};

                if (!bo || useDis < minOutsideDist)
                {
                    bestOutside    = curLP;
                    minOutsideDist = useDis;
                    bo             = true;
                }
            }
        }

        if (bi)
        {
            return bestInside;
        }
        else if (bo)
        {
            return bestOutside;
        }

        return out;
    }

    //========================================
    //!\brief Used to find a \ref LaneSegment from a vehicle state.
    //!\param[in] point  The GSP point
    //!\return Returns information about distance to center of lane for the point
    //!        and distance from start of the segment along the lane to the point
    //!\attention Only valid, if the \ref CarriageWay was created with reference
    //!           GPS(0,0,0).
    //----------------------------------------
    LanePosition getLanePosition(const VehicleState& point) const
    {
        PositionWgs84 p;
        p.transformFromTangentialPlane2d(
            point.getRelativePosition().getX(), point.getRelativePosition().getY(), point.getOriginWgs84());
        p.setCourseAngleInRad(point.getCourseAngle());

        return getLanePosition(p);
    }

    //========================================
    //!\brief Calculates the LanePosition of system given relative to a reference system
    //!\param[in] reference             The system of the reference position (e.g. the ego
    //!                                 position obtained from getLanePosition).
    //!\param[in] relativePosition      The relative system for which the LanePosition
    //!                                 shall be obtained
    //!\param[out] insideReferenceLane  \c true, if the relative system is within the same
    //!                                 lane as the reference system, \c false otherwise
    //!\param[out] success              \c true, if the calculation was successful,
    //!                                 \c false otherwise.
    //----------------------------------------
    LanePosition getReferenceLanePosition(const LanePosition& reference,
                                          const TransformationMatrix2d<float>& relativePosition,
                                          bool& insideReferenceLane,
                                          bool& success) const
    {
        LanePosition out{};
        success             = false;
        insideReferenceLane = false;

        PositionWgs84 objectPosition;
        objectPosition.transformFromTangentialPlaneWithHeading2d(relativePosition.getPositionVector().getX(),
                                                                 relativePosition.getPositionVector().getY(),
                                                                 reference.gpsPosition);
        objectPosition.setCourseAngleInRad(reference.gpsPosition.getCourseAngleInRad()
                                           + relativePosition.getRotationMatrix().getAngle());

        if (reference.laneSegment)
        {
            // extract all segments with the same lane id as the reference lane
            uint64_t refLaneId{reference.laneSegment->getParent()->getLaneId()};
            std::vector<LaneSegmentPtr> validLaneSegments;

            for (const auto& way : m_ways)
            {
                for (const auto& cwsme : way->getCwSegmentsMap())
                {
                    for (const auto& lme : cwsme.second->getLanesMap())
                    {
                        for (const auto& lsme : lme.second->getLaneSegmentsMap())
                        {
                            if (lsme.second->getParent()->getLaneId() == refLaneId)
                            {
                                validLaneSegments.push_back(lsme.second);
                            }
                        } // for all lane segments
                    } // for all lanes
                } // for all carriage way segments
            } // for all carriage ways

            if (!validLaneSegments.empty())
            {
                // find the best suitable
                LanePosition bestInside{};
                LanePosition bestOutside{};
                float minOutsideDist{0.0F};
                bool bi{false};
                bool bo{false};
                for (const auto& curSeg : validLaneSegments)
                {
                    bool inside{false};
                    LanePosition curLP{calculateLanePosition(curSeg, objectPosition, inside)};

                    if (inside && (!bi || std::abs(curLP.lateralDistance) < std::abs(bestInside.lateralDistance)))
                    {
                        bestInside = curLP;
                        bi         = true;
                    }
                    else if (!inside)
                    {
                        const float startDis{std::abs(curLP.longitudinalDistance)};
                        const float endDis{std::abs(curLP.longitudinalDistance - curSeg->getLength())};
                        const float useDis{startDis < endDis ? startDis : endDis};

                        if (!bo || useDis < minOutsideDist)
                        {
                            bestOutside    = curLP;
                            minOutsideDist = useDis;
                            bo             = true;
                        }
                    }
                } // for all valid segments

                if (bi || bo)
                {
                    LanePosition lP{bi ? bestInside : bestOutside};
                    const float length{bi ? bestInside.longitudinalDistance : 0.0F};

                    if (std::abs(lP.lateralDistance) <= lP.laneSegment->getWidth(length) / 2)
                    {
                        insideReferenceLane = true;
                    }

                    // count length to front
                    LaneSegmentPtr startSeg{reference.laneSegment};
                    float lengthCount{0};
                    if (relativePosition.getPositionVector().getX() >= 0)
                    {
                        while (startSeg && lengthCount <= 300)
                        {
                            if (lP.laneSegment == startSeg)
                            {
                                lP.longitudinalDistance
                                    = lengthCount + lP.longitudinalDistance - reference.longitudinalDistance;
                                success = true;
                                return lP;
                            }

                            lengthCount += startSeg->getLength();
                            startSeg = startSeg->getNext();
                        }
                    }
                    else
                    {
                        while (startSeg && length >= -300)
                        {
                            if (lP.laneSegment == startSeg)
                            {
                                lP.longitudinalDistance
                                    = lengthCount + lP.longitudinalDistance - reference.longitudinalDistance;
                                success = true;
                                return lP;
                            }

                            startSeg = startSeg->getPrevious();
                            if (!startSeg)
                            {
                                break;
                            }
                            lengthCount -= startSeg->getLength();
                        } // while
                    } // else
                } // if
            } // if
        } // if
        return out;
    }

    //========================================
    //! \brief Calculates a transformationMatrix relative to the reference system by giving
    //!        a longitudinal and lateral distance
    //!\param[in]  reference             The reference LanePosition
    //!\param[in]  longitudinalDistance  The longitudinal distance relative from the reference
    //!\param[in]  lateralDistance       The lateral distance relative from the reference
    //!\param[out] success               \c true, if successful, \c false otherwise
    //----------------------------------------
    TransformationMatrix2d<float> getRelativePosition(const LanePosition& reference,
                                                      const float& longitudinalDistance,
                                                      const float& lateralDistance,
                                                      bool& success) const
    {
        success = false;

        // find lane segment
        float ldistance{longitudinalDistance + reference.longitudinalDistance};
        LaneSegmentPtr startSeg{reference.laneSegment};

        if (ldistance > 0)
        {
            while (startSeg && ldistance > startSeg->getLength())
            {
                ldistance -= startSeg->getLength();
                startSeg = startSeg->getNext();
            }
        }
        else
        {
            while (startSeg && ldistance < 0)
            {
                startSeg = startSeg->getPrevious();
                if (!startSeg)
                {
                    break;
                }
                ldistance += startSeg->getLength();
            }
        }

        if (startSeg)
        {
            //create gps position for point
            PositionWgs84 objectPosition;
            objectPosition.transformFromTangentialPlaneWithHeading2d(
                ldistance, lateralDistance, startSeg->getStartPoint().getWgsPoint());

            // create matrix relative to reference
            double x;
            double y;
            objectPosition.transformToTangentialPlaneWithHeading2d(reference.gpsPosition, x, y);

            Vector2<float> position{Vector2<float>(float(x), float(y))};
            float angle{
                static_cast<float>(objectPosition.getCourseAngleInRad() - reference.gpsPosition.getCourseAngleInRad())};

            success = true;
            return TransformationMatrix2d<float>(angle, position);
        }
        else
        {
            success = false;
            return TransformationMatrix2d<float>();
        }
    }

private:
    LanePosition
    calculateLanePosition(const LaneSegmentPtr& segment, const PositionWgs84& point, bool& insideIntersection) const
    {
        double xs;
        double ys;
        point.transformToTangentialPlaneWithHeading2d(segment->getStartPoint().getWgsPoint(), xs, ys);
        Vector2<float> p{Vector2<float>(float(xs), float(ys))};
        Line2d<float> line{Line2d<float>(Vector2<float>(), segment->getEndOffset())};
        Vector2<float> lineDir{line.getDiff().getNormalized()};
        Vector2<float> perpDir{Vector2<float>(-lineDir.getY(), lineDir.getX())};
        Line2d<float> perp{Line2d<float>(p + perpDir * 15, p - perpDir * 15)}; // 15m in both directions

        Vector2<float> intersection;
        Line2d<float>::IntersectingType ist{line.isIntersecting(perp, &intersection)};

        insideIntersection = (ist == Line2d<float>::IntersectingType::LineIntersecting);

        LanePosition out{};
        out.laneSegment          = segment;
        out.transformationMatrix = TransformationMatrix2d<float>(
            static_cast<float>(point.getCourseAngleInRad()
                               - segment->getStartPoint().getWgsPoint().getCourseAngleInRad()),
            p);
        out.lateralDistance      = p.getY();
        out.longitudinalDistance = p.getX();
        out.gpsPosition          = point;

        return out;
    }

private:
    CarriageWays m_ways;
}; // class LaneHandlerTemplate

//==============================================================================

using LaneHandler        = LaneHandlerTemplate<LaneSegment>;
using LaneHandlerFor6970 = LaneHandlerTemplate<LaneSegmentIn6970>;
using LaneHandlerFor6972 = LaneHandlerTemplate<LaneSegmentIn6972>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

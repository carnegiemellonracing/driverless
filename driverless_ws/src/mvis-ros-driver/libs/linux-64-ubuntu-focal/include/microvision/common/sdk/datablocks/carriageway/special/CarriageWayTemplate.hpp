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
//! \date Dec 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWaySegmentTemplate.hpp>
#include <microvision/common/sdk/LaneType.hpp>
#include <microvision/common/sdk/io.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWayTemplate represents one surface of a road and has a unique identifier.
//!
//! The identifier is a combination of the \ref CarriageWayType of  the road and a
//! number (e.g. A for \ref CarriageWayType motorway and 1 represents A1).
//!
//! In addition each CarriageWayTemplate holds a list of segments. Within one segment,
//! the number of lanes is constant. If there are preceding and following segments,
//! these segments are linked against each other. It is therefore possible to
//! store multiple linked lists of segments within on CarriageWayTemplate (e.g. for different
//! driving directions or if there are gaps between segments).
//!
//! A CarriageWay is always the highest representation of a road. The segmentation
//! is as following:
//!
//!\ref CarriageWayTemplate \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentInTemplate
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create() will return
//! a shared pointer to a new CarriageWayTemplate.
//!
//!\sa CarriageWaySegmentTemplate \sa LaneTemplate \sa LaneSegmentInTemplate
//------------------------------------------------------------------------------
template<typename TLaneSegment>
class CarriageWayTemplate final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;

public:
    // Shared/weak pointer to this class.
    using Ptr     = std::shared_ptr<CarriageWayTemplate>;
    using WeakPtr = std::weak_ptr<CarriageWayTemplate>;

    // Map with carriage way segments.
    using CarriageWaySegmentTemplatePtr    = std::shared_ptr<CarriageWaySegmentTemplate<TLaneSegment>>;
    using CarriageWaySegmentTemplatePtrMap = std::unordered_map<uint64_t, CarriageWaySegmentTemplatePtr>;

public:
    //========================================
    //!\brief Default constructor.
    //!
    //! Initializes all elements to 0
    //!\return A shared pointer to the created CarriageWayTemplate.
    //----------------------------------------
    static Ptr create() { return Ptr(new CarriageWayTemplate()); }

    //========================================
    //!\brief Constructor
    //!\param[in] id          A unique id for this CarriageWayTemplate for internal identification
    //!\param[in] nationalId  The national number of the road (e.g. 1 for B1 or A1)
    //!\param[in] type        The type of the road (motorway, trunk, ...)
    //!\param[in] segments    A map holding all CarriageWaySegmentTemplates with unique identifiers
    //----------------------------------------
    static Ptr create(const uint64_t& id,
                      const uint16_t& nationalId,
                      const CarriageWayType& type,
                      const CarriageWaySegmentTemplatePtrMap& segments = CarriageWaySegmentTemplatePtrMap())
    {
        return Ptr(new CarriageWayTemplate(id, nationalId, type, segments));
    }

private: // Constructors.
    //! Private constructor called by \ref create() \sa create.
    CarriageWayTemplate() = default;

    //! Private constructor called by \ref create() \sa create.
    CarriageWayTemplate(const uint64_t& id,
                        const uint16_t& nationalId,
                        const CarriageWayType& type,
                        const CarriageWaySegmentTemplatePtrMap& segments = CarriageWaySegmentTemplatePtrMap())
      : m_id{id}, m_nationalId{nationalId}, m_type{type}, m_cwSegmentsMap{segments}
    {}

    //========================================
    // getter
    //========================================
public:
    //!\return The internally used unique id of the road.
    uint64_t getId() const { return m_id; }

    //!\return The national id of this road.
    uint16_t getNationalId() const { return m_nationalId; }

    //!\returns The type of the road.
    CarriageWayType getType() const { return m_type; }

    //========================================
    //!\brief Returns all segments of the road.
    //!\return A map holding pointers to all segments of the road which can be
    //!        accessed by a unique id.
    //----------------------------------------
    const CarriageWaySegmentTemplatePtrMap& getCwSegmentsMap() const { return m_cwSegmentsMap; }

    //========================================
    //!\brief Returns a bounding rectangle of the way which is currently not the
    //!       minimal one, but the one aligned to the north vector
    //----------------------------------------
    BoundingRectangle getBoundingBox() const { return m_boundingBox; }

    //========================================
    // getter
    //========================================
public:
    //! Sets the id of the CarriageWay.
    void setId(const uint64_t& id) { m_id = id; }

    //! Sets the type of the road to the given parameter.
    void setType(const CarriageWayType& type) { m_type = type; }

    //!\brief Sets all road segments to the given list.
    void setCwSegmentsMap(const CarriageWaySegmentTemplatePtrMap& segments) { m_cwSegmentsMap = segments; }

public:
    //========================================
    //!\brief Inserts a new segment to the list for this CarriageWay.
    //!\return \c True if the inserting is possible (id not already in the list)
    //!        and \c false if an inserting is not possible
    //----------------------------------------
    bool insert(CarriageWaySegmentTemplatePtr segment)
    {
        // inserts the segment to the map with its unique id and checks if another
        // segment with the same id already exists.
        return (m_cwSegmentsMap.insert(
                    typename CarriageWaySegmentTemplatePtrMap::value_type(segment->getId(), segment)))
            .second;
    }

    //========================================
    //!\brief Resolves the connections between segments, lanes, etc.
    //!
    //! This function has to be called once the segments are all filled. It
    //! generates pointers between LaneSegments, Lanes, CarriageWaySegments as
    //!  well as pointers to the parent object.
    //----------------------------------------
    void resolveConnections(const Ptr& ptrToMe)
    {
        resolveCwSegmentConnections(ptrToMe);
        resolveLaneConnections();
        resolveLaneSegmentConnections();
    }

    //========================================
    //! \brief Compares two shared pointers to carriage ways for equality.
    //!
    //! \param[in] lhs  Left hand side for comparison.
    //! \param[in] rhs  Right hand side for comparison.
    //! \return \c True, if both pointers are null or the two carriage ways are equal, \c false otherwise.
    //----------------------------------------
    static bool areSharedPtrObjectsEqual(const Ptr& lhs, const Ptr& rhs)
    {
        if ((lhs != nullptr) && (rhs != nullptr))
        {
            // Both pointers have objects -> compare them.
            return *lhs == *rhs;
        }
        else
        {
            // At least one of the pointers is null -> equal only if both pointers are null.
            return (lhs == nullptr) && (rhs == nullptr);
        }
    }

private:
    //========================================

    void resolveCwSegmentConnections(const Ptr& ptrToMe)
    {
        // 1. Create all connections between carriageway segments, since these
        // are required for connections between lanes and lane segments
        for (auto& cwsme : m_cwSegmentsMap)
        {
            // pointer to segment (the object itself has to be edited, not a copy)
            CarriageWaySegmentTemplatePtr cws = cwsme.second;
            if (cws)
            {
                cws->setParent(ptrToMe);

                // link next element if id not 0 and existing
                if ((cws->getNextId() != 0))
                {
                    const auto itf = m_cwSegmentsMap.find(cws->getNextId());
                    if (itf != m_cwSegmentsMap.end())
                    {
                        cws->setNext(itf->second);
                    }
                }

                // link previous element if id not 0 and existing
                if ((cws->getPrevId() != 0))
                {
                    const auto itf = m_cwSegmentsMap.find(cws->getPrevId());
                    if (itf != m_cwSegmentsMap.end())
                    {
                        cws->setPrevious(itf->second);
                    }
                }
            } // if cws
        } // for all cws map entries
    }

    //========================================

    void resolveLaneConnections()
    {
        // 2. Create all connections between lanes
        // now the pointers to previous and next segment between CarriageWaySegments
        // are all resolved

        for (auto& cwsme : m_cwSegmentsMap)
        {
            CarriageWaySegmentTemplatePtr cws = cwsme.second;
            if (cws)
            {
                for (auto& lme : cws->getLanesMap())
                {
                    typename CarriageWaySegmentTemplate<TLaneSegment>::LaneTemplatePtr lane = lme.second;
                    if (lane)
                    {
                        resolveLaneConnections(cws, lane);
                    } // if lane
                } // for laneIter
            } // if cws
        } // for cwsmIter
    }

    //========================================

    void resolveLaneConnections(CarriageWaySegmentTemplatePtr cws, typename LaneTemplate<TLaneSegment>::Ptr lane)
    {
        lane->setParent(cws);

        // link left neighbour
        if (lane->getLeftLaneId() != 0)
        {
            const auto neighborLaneIter = cws->getLanesMap().find(lane->getLeftLaneId());
            if (neighborLaneIter != cws->getLanesMap().end())
            {
                lane->setLeft(neighborLaneIter->second);
            }
        }

        // link right neighbour
        if (lane->getRightLaneId() != 0)
        {
            const auto neighborLaneIter = cws->getLanesMap().find(lane->getRightLaneId());
            if (neighborLaneIter != cws->getLanesMap().end())
            {
                lane->setRight(neighborLaneIter->second);
            }
        }

        // link next lane (always in next CarriageWaySegmentIn6970)
        if (cws->hasNext() && (lane->getNextLaneId() != 0))
        {
            const auto neighborLaneIter = cws->getNext()->getLanesMap().find(lane->getNextLaneId());
            if (neighborLaneIter != cws->getNext()->getLanesMap().end())
            {
                lane->setNext(neighborLaneIter->second);
            }
        }

        // link previous lane (always in previous CarriageWaySegmentIn6970)
        if (cws->hasPrevious() && (lane->getPrevLaneId() != 0))
        {
            const auto neighborLaneIter = cws->getPrevious()->getLanesMap().find(lane->getPrevLaneId());
            if (neighborLaneIter != cws->getPrevious()->getLanesMap().end())
            {
                lane->setPrevious(neighborLaneIter->second);
            }
        } // if
    }

    //========================================

    void resolveLaneSegmentConnections()
    {
        // 3. Create all connections between LaneSegments
        // now the pointers to previous and next lane are all resolved
        for (auto& cwsme : m_cwSegmentsMap)
        {
            CarriageWaySegmentTemplatePtr cws = cwsme.second;
            if (cws)
            {
                for (auto& lme : cws->getLanesMap())
                {
                    typename CarriageWaySegmentTemplate<TLaneSegment>::LaneTemplatePtr lane = lme.second;
                    if (lane)
                    {
                        resolveLaneSegmentConnections(cws, lane);
                    }
                }

                // set bounding box for CARRIAGE_WAY from child bounds
                m_boundingBox.expand(cws->getBoundingBox());
            }
        }
    }

    //========================================

    void resolveLaneSegmentConnections(CarriageWaySegmentTemplatePtr cws, typename LaneTemplate<TLaneSegment>::Ptr lane)
    {
        for (auto& lsme : lane->getLaneSegmentsMap())
        {
            typename LaneTemplate<TLaneSegment>::LaneSegmentPtr laneSeg = lsme.second;

            if (laneSeg)
            {
                laneSeg->setParent(lane);

                //========================================
                // next segment
                if ((!laneSeg->isNextInNewSeg()) && (laneSeg->getNextId() != 0))
                { // link next segment if in same CarriageWaySegmentIn6970
                    const auto laneSegIter = lane->getLaneSegmentsMap().find(laneSeg->getNextId());
                    if (laneSegIter != lane->getLaneSegmentsMap().end())
                    {
                        laneSeg->setNext(laneSegIter->second);
                    }
                } // if
                else if (lane->hasNext() && (laneSeg->getNextId() != 0))
                { // if in new CarriageWaySegmentIn6970
                    const auto laneSegIter = lane->getNext()->getLaneSegmentsMap().find(laneSeg->getNextId());
                    if (laneSegIter != lane->getNext()->getLaneSegmentsMap().end())
                    {
                        laneSeg->setNext(laneSegIter->second);
                    }
                } // else if

                //========================================
                // previous segment
                if ((!laneSeg->isPrevInNewSeg()) && (laneSeg->getPrevId() != 0))
                { // // link previous segment if in same CarriageWaySegmentIn6970
                    const auto laneSegIter = lane->getLaneSegmentsMap().find(laneSeg->getPrevId());
                    if (laneSegIter != lane->getLaneSegmentsMap().end())
                    {
                        laneSeg->setPrevious(laneSegIter->second);
                    }
                }
                else if (lane->hasPrevious() && (laneSeg->getPrevId() != 0))
                { // if in new CarriageWaySegmentIn6970
                    const auto laneSegIter = lane->getPrevious()->getLaneSegmentsMap().find(laneSeg->getPrevId());
                    if (laneSegIter != lane->getPrevious()->getLaneSegmentsMap().end())
                    {
                        laneSeg->setPrevious(laneSegIter->second);
                    }
                } // else if

                //========================================
                // link left
                if (lane->hasLeft() && (laneSeg->getLeftId() != 0))
                {
                    const auto laneSegIter = lane->getLeft()->getLaneSegmentsMap().find(laneSeg->getLeftId());
                    if (laneSegIter != lane->getLeft()->getLaneSegmentsMap().end())
                    {
                        laneSeg->setLeft(laneSegIter->second);
                    }
                } // if

                //========================================
                // link right
                if (lane->hasRight() && (laneSeg->getRightId() != 0))
                {
                    const auto laneSegIter = lane->getRight()->getLaneSegmentsMap().find(laneSeg->getRightId());
                    if (laneSegIter != lane->getRight()->getLaneSegmentsMap().end())
                    {
                        laneSeg->setRight(laneSegIter->second);
                    }
                } // if

                laneSeg->updateProperties();

                // set bounding box for LANE from child bounds
                lane->m_boundingBox.expand(laneSeg->getBoundingBox());
            } // if laneSeg
        } // for all lane segment map entries

        // set bounding box for CARRIAGE_WAY_SEGMENT from child bounds
        cws->m_boundingBox.expand(lane->getBoundingBox());
    }

    //========================================
    // Members.
    //========================================
protected:
    //========================================
    uint64_t m_id{0}; //!< The unique id of this Carriageway.
    uint16_t m_nationalId{0}; //!< The national Id of the CarriageWay (e.g. 255 for street A255).

    CarriageWayType m_type{CarriageWayType::Unclassified}; //!< The type of the CarriageWay.

    CarriageWaySegmentTemplatePtrMap m_cwSegmentsMap{}; //!< The segments of this CarriageWay.

    BoundingRectangle m_boundingBox{};
}; // class CarriageWayTemplate

//==============================================================================

//==============================================================================
//! \brief Compares two carriage ways for equality.
//!
//! \param[in] lhs  Left hand side for comparison.
//! \param[in] rhs  Right hand side for comparison.
//! \return \c True, if the two carriage ways are equal, \c false otherwise.
//------------------------------------------------------------------------------
template<typename TLaneSegment>
inline bool operator==(const CarriageWayTemplate<TLaneSegment>& lhs, const CarriageWayTemplate<TLaneSegment>& rhs)
{
    const bool ok = (lhs.getNationalId() == rhs.getNationalId()) && (lhs.getType() == rhs.getType())
                    && (lhs.getBoundingBox() == rhs.getBoundingBox());

    if (!ok)
    {
        return false;
    }

    for (auto& cwsme : lhs.getCwSegmentsMap())
    {
        if (std::find_if(
                rhs.getCwSegmentsMap().begin(),
                rhs.getCwSegmentsMap().end(),
                [&cwsme](const typename CarriageWayTemplate<TLaneSegment>::CarriageWaySegmentTemplatePtrMap::value_type&
                             cwsp) {
                    return CarriageWaySegmentTemplate<TLaneSegment>::areEqualRecursive(*cwsme.second, *cwsp.second);
                })
            == rhs.getCwSegmentsMap().end())
        {
            return false;
        }
    }

    return true;
}

//==============================================================================
//! \brief Compares two carriage ways for inequality.
//!
//! \param[in] lhs  Left hand side for comparison.
//! \param[in] rhs  Right hand side for comparison.
//! \return \c True, if the two carriage ways are not equal, \c false otherwise.
//------------------------------------------------------------------------------
template<typename TLaneSegment>
inline bool operator!=(const CarriageWayTemplate<TLaneSegment>& lhs, const CarriageWayTemplate<TLaneSegment>& rhs)
{
    return !(lhs == rhs);
}

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

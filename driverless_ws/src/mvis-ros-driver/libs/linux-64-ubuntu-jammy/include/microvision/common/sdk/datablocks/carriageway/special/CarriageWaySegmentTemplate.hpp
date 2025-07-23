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

#include <microvision/common/sdk/datablocks/carriageway/special/LaneTemplate.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

// Forward declarations for parent reference.
template<typename TLaneSegment>
class CarriageWayTemplate;

//==============================================================================
//!\brief A CarriageWaySegmentTemplate represents a single segment of a \ref CarriageWayTemplate.
//!
//! Each CarriageWaySegmentTemplate has a unique id within the parent \ref CarriageWayTemplate.
//!
//! A \ref CarriageWayTemplate holds a constant number of lanes of type LaneTemplate. The segmentation of a whole
//! \ref CarriageWayTemplate is as following:
//!
//!\ref CarriageWayTemplate \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentInTemplate
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create() will return
//! a shared pointer to a new CarriageWaySegmentTemplate.
//!
//!\sa CarriageWayTemplate \sa LaneTemplate \sa LaneSegmentInTemplate
//------------------------------------------------------------------------------
template<typename TLaneSegment>
class CarriageWaySegmentTemplate final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;
    template<typename>
    friend class CarriageWayTemplate;

public:
    // Shared/weak pointer to this class.
    using Ptr     = std::shared_ptr<CarriageWaySegmentTemplate>;
    using WeakPtr = std::weak_ptr<CarriageWaySegmentTemplate>;

    // Shared/weak pointer to parent classes.
    using CarriageWayTemplatePtr     = std::shared_ptr<CarriageWayTemplate<TLaneSegment>>;
    using CarriageWayTemplateWeakPtr = std::weak_ptr<CarriageWayTemplate<TLaneSegment>>;

    // Map with lanes.
    using LaneTemplatePtr    = std::shared_ptr<LaneTemplate<TLaneSegment>>;
    using LaneTemplatePtrMap = std::unordered_map<uint8_t, LaneTemplatePtr>;

public:
    //========================================
    //!\brief Creates a CarriageWaySegmentTemplate.
    //!
    //! Initializes all elements to 0
    //!\return A shared pointer to the created CarriageWaySegmentTemplate.
    //----------------------------------------
    static Ptr create() { return Ptr(new CarriageWaySegmentTemplate()); }

    //========================================
    //!\brief Creates a CarriageWaySegmentTemplate.
    //!\param[in] parent  The pointer to the parent CarriageWay.
    //!\return A shared pointer to the created CarriageWaySegmentTemplate.
    //----------------------------------------
    static Ptr create(CarriageWayTemplatePtr parent) { return Ptr(new CarriageWaySegmentTemplate(parent)); }

    //========================================
    //!\brief Creates a CarriageWaySegmentTemplate.
    //!\param[in] id      The unique id within the parent CarriageWay.
    //!\param[in] parent  The pointer to the parent CarriageWay.
    //!\return A shared pointer to the created CarriageWaySegmentTemplate.
    //----------------------------------------
    static Ptr create(const uint64_t& id, CarriageWayTemplatePtr parent)
    {
        return Ptr(new CarriageWaySegmentTemplate(id, parent));
    }

    //========================================
    //!\brief Creates a CarriageWaySegmentTemplate.
    //!\param[in] id                 The unique id within the parent CarriageWay
    //!\param[in] parent             The pointer to the parent CarriageWay
    //!\param[in] nextSegmentId      The unique id of the next segment within the same CarriageWay
    //!\param[in] previousSegmentId  The unique id of the previous segment within the same CarriageWay
    //!\return A shared pointer to the created CarriageWaySegmentTemplate
    //----------------------------------------
    static Ptr create(const uint64_t& id,
                      CarriageWayTemplatePtr parent,
                      const uint64_t nextSegmentId,
                      const uint64_t previousSegmentId)
    {
        return Ptr(new CarriageWaySegmentTemplate(id, parent, nextSegmentId, previousSegmentId));
    }

    static Ptr copy(const Ptr& other)
    {
        Ptr out(new CarriageWaySegmentTemplate());
        out->setId(other->getId());
        out->setPrevId(other->getPrevId());
        out->setNextId(other->getNextId());

        return out;
    }

private:
    //========================================
    //!\brief Private constructor called by \ref create().
    //!\sa create
    //----------------------------------------
    CarriageWaySegmentTemplate() = default;

    //========================================
    //!\brief Private constructor called by \ref create().
    //!\sa create
    //----------------------------------------
    CarriageWaySegmentTemplate(CarriageWayTemplatePtr parent) : m_parent(parent) {}

    //========================================
    //!\brief Private constructor called by \ref create().
    //!\sa create
    //----------------------------------------
    CarriageWaySegmentTemplate(const uint64_t id, CarriageWayTemplatePtr parent) : m_id(id), m_parent(parent) {}

    //========================================
    //!\brief Private constructor called by \ref create().
    //!\sa create
    //----------------------------------------
    CarriageWaySegmentTemplate(const uint64_t id,
                               CarriageWayTemplatePtr parent,
                               const uint64_t nextSegmentId,
                               const uint64_t previousSegmentId)
      : m_id(id), m_parent(parent), m_nextId(nextSegmentId), m_prevId(previousSegmentId)
    {}

    //========================================
    // getter
    //========================================
public:
    //!\returns The id of the segment.
    uint64_t getId() const { return m_id; }

    //!\returns The pointer to the parent CarriageWayTemplate (0 if not set).
    const CarriageWayTemplatePtr getParent() const { return m_parent.lock(); }

    uint64_t getNextId() const { return m_nextId; }
    uint64_t getPrevId() const { return m_prevId; }

    //!\return The number of lanes within that segment.
    uint8_t getNbOfLanes() const { return static_cast<uint8_t>(m_lanesMap.size()); }

    //!\returns a constant reference to the map of child lanes.
    const LaneTemplatePtrMap& getLanesMap() const { return m_lanesMap; }

    //!\returns A reference to the map of child lanes.
    LaneTemplatePtrMap& getLanesMap() { return m_lanesMap; }

    //!\returns The pointer to the next segment (0 if there is none).
    const Ptr getNext() const { return m_nextSegment.lock(); }

    //!\returns the pointer to the previous segment (0 if there is none).
    const Ptr getPrevious() const { return m_prevSegment.lock(); }

    //! Returns true if the segment has a following segment, false otherwise.
    bool hasNext() const { return (m_nextSegment.lock() != nullptr); }

    //! Returns true if the segment has a preceding segment, false otherwise.
    bool hasPrevious() const { return (m_prevSegment.lock() != nullptr); }

    //!\returns The length of this segment.
    //float getLength() const;

    //!\returns A bounding rectangle of the way which is currently not the
    //! minimal one, but the one aligned to the north vector
    BoundingRectangle getBoundingBox() const { return m_boundingBox; }

    //========================================
    // setter
    //========================================
public:
    void setId(const uint64_t& id) { m_id = id; }

    //! Sets the pointer to the parent \ref CarriageWayTemplate.
    void setParent(const CarriageWayTemplatePtr& parent) { m_parent = parent; }

    void setNextId(const uint64_t& nextId) { m_nextId = nextId; }
    void setPrevId(const uint64_t& prevId) { m_prevId = prevId; }

    //! Sets the child lanes to the given map.
    void setLanesMap(const LaneTemplatePtrMap& lanesMap) { m_lanesMap = lanesMap; }

    //! Sets the pointer to the next segment.
    void setNext(const Ptr& next)
    {
        m_nextSegment = next;
        m_nextId      = next ? next->getId() : 0;
    }

    //! Sets the pointer to the previous segment.
    void setPrevious(const Ptr& previous)
    {
        m_prevSegment = previous;
        m_prevId      = previous ? previous->getId() : 0;
    }

    //	//! Sets the length of this segment (length along center line).
    //	void setLength(const float& length);

    //========================================
    // equality functions
    //========================================
public:
    static bool arePtrsEqualRecursive(const typename CarriageWaySegmentTemplate<TLaneSegment>::Ptr& lhs,
                                      const typename CarriageWaySegmentTemplate<TLaneSegment>::Ptr& rhs)
    {
        if ((!lhs) && (!rhs))
        {
            return true;
        }

        if ((!lhs) || (!rhs))
        {
            return false;
        }

        return areEqualNonRecursive(*lhs, *rhs);
    }

    //========================================

    static bool areEqualRecursive(const CarriageWaySegmentTemplate<TLaneSegment>& lhs,
                                  const CarriageWaySegmentTemplate<TLaneSegment>& rhs)
    {
        const bool ok = (lhs.getNbOfLanes() == rhs.getNbOfLanes()) && (lhs.getBoundingBox() == rhs.getBoundingBox())
                        && arePtrsEqualRecursive(lhs.getNext(), rhs.getNext())
                        && arePtrsEqualRecursive(lhs.getPrevious(), rhs.getPrevious());

        if (!ok)
        {
            return false;
        }

        for (auto& lme : lhs.getLanesMap())
        {
            if (std::find_if(
                    rhs.getLanesMap().begin(),
                    rhs.getLanesMap().end(),
                    [&lme](
                        const typename CarriageWaySegmentTemplate<TLaneSegment>::LaneTemplatePtrMap::value_type& lsp) {
                        return LaneTemplate<TLaneSegment>::areEqualRecursive(*lme.second, *lsp.second);
                    })
                == rhs.getLanesMap().end())
            {
                return false;
            }
        }

        return true;
    }

    //========================================

    static bool areEqualNonRecursive(const CarriageWaySegmentTemplate<TLaneSegment>& lhs,
                                     const CarriageWaySegmentTemplate<TLaneSegment>& rhs)
    {
        const bool ok = (lhs.getNbOfLanes() == rhs.getNbOfLanes()) && (lhs.getBoundingBox() == rhs.getBoundingBox());

        if (!ok)
        {
            return false;
        }

        for (auto& lme : lhs.getLanesMap())
        {
            if (std::find_if(
                    rhs.getLanesMap().begin(),
                    rhs.getLanesMap().end(),
                    [&lme](
                        const typename CarriageWaySegmentTemplate<TLaneSegment>::LaneTemplatePtrMap::value_type& lsp) {
                        return LaneTemplate<TLaneSegment>::areEqualRecursive(*lme.second, *lsp.second);
                    })
                == rhs.getLanesMap().end())
            {
                return false;
            }
        }

        return true;
    }

public:
    //========================================

    void updateConnectionIds(const Ptr& reference, const bool override = true)
    {
        if (reference)
        {
            if (m_nextId == 0 || override)
            {
                m_nextId = reference->getNextId();
            }
            if (m_prevId == 0 || override)
            {
                m_prevId = reference->getPrevId();
            }
        }
    }

    //========================================

    uint8_t getNextFreeKey() const
    {
        uint8_t maxKey = 0;

        for (auto& lme : m_lanesMap)
        {
            if (lme.first > maxKey)
            {
                maxKey = lme.first;
            }
        } // for

        return (++maxKey);
    }

    //========================================
    //! Inserts a single lane to the map of child lanes.
    bool insert(typename LaneTemplate<TLaneSegment>::Ptr lane)
    {
        // inserts the lane to the map with its unique id and checks if another
        // lane with the same id already exists.
        return (m_lanesMap.insert(typename LaneTemplatePtrMap::value_type(lane->getId(), lane))).second;
    }

    //========================================

    void cleanIds()
    {
        m_nextId = hasNext() ? getNext()->getId() : 0;
        m_prevId = hasPrevious() ? getPrevious()->getId() : 0;

        for (auto& lme : m_lanesMap)
        {
            lme.second->cleanIds();
        }
    }

    //========================================
    // Members
    //========================================
protected:
    uint64_t m_id{0}; //!< The unique id of this CarriagewaySegment.
    CarriageWayTemplateWeakPtr m_parent{};
    uint64_t m_nextId{0}; //!< The ID of the following segment.
    uint64_t m_prevId{0}; //!< The ID of the previous segment.

    LaneTemplatePtrMap m_lanesMap{}; //!< The lanes of this segment.

    WeakPtr m_nextSegment{};
    WeakPtr m_prevSegment{};

    BoundingRectangle m_boundingBox{};
}; // class CarriageWaySegmentTemplate

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

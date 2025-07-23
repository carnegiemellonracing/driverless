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
//! \date Dec 12, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/LaneType.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/RegisteredImporter.hpp>

#include <unordered_map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

// Forward declarations.
template<typename TLaneSegment>
class CarriageWaySegmentTemplate;

template<typename TLaneSegment>
class CarriageWayTemplate;

//==============================================================================
//!\brief A LaneTemplate represents a lane of a specific type within a \ref CarriageWaySegmentTemplate
//!
//! Each LaneTemplate has a unique id within the parent \ref CarriageWaySegmentTemplate
//!
//! A LaneTemplate holds a list of for example LaneSegmentIn6972 segments as well as pointers to preceding,
//! following and neighboring Lanes.
//!
//! The segmentation of a whole road is as following:
//!
//!\ref CarriageWayTemplate \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneTemplate \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6972
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create will return
//! a shared pointer to a new LaneTemplate.
//!
//!\sa CarriageWayTemplate \sa CarriageWaySegmentTemplate \sa LaneSegmentIn6972
//------------------------------------------------------------------------------
template<typename TLaneSegment>
class LaneTemplate final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;
    friend class CarriageWayTemplate<TLaneSegment>;

public:
    // Shared/weak pointer to this class.
    using Ptr     = std::shared_ptr<LaneTemplate>;
    using WeakPtr = std::weak_ptr<LaneTemplate>;

    // Shared/weak pointer to parent classes.
    using CarriageWaySegmentTemplatePtr     = std::shared_ptr<CarriageWaySegmentTemplate<TLaneSegment>>;
    using CarriageWaySegmentTemplateWeakPtr = std::weak_ptr<CarriageWaySegmentTemplate<TLaneSegment>>;

    // Map with lane segments.
    using LaneSegmentPtr    = std::shared_ptr<TLaneSegment>;
    using LaneSegmentPtrMap = std::unordered_map<uint64_t, LaneSegmentPtr>;

private: // constructors
    //========================================
    //! Private constructor called by \ref create() \sa create.
    LaneTemplate() = default;

    //========================================
    //! Private constructor called by \ref create() \sa create.
    explicit LaneTemplate(CarriageWaySegmentTemplateWeakPtr parent) : m_parent(parent) {}

    //========================================
    //! Private constructor called by \ref create() \sa create.
    LaneTemplate(const uint8_t& id,
                 const uint64_t laneId,
                 const LaneType& type,
                 CarriageWaySegmentTemplateWeakPtr parent,
                 const uint8_t& nextLaneId,
                 const uint8_t& previousLaneId,
                 const uint8_t& leftLaneId,
                 const uint8_t& rightLaneId);

public:
    //========================================
    //!\brief Default constructor
    //!
    //! Initializes all members to 0
    //!\return A shared pointer to the created LaneTemplate.
    //----------------------------------------
    static Ptr create() { return Ptr(new LaneTemplate()); }

    //========================================
    //!\brief Constructor
    //!\param[in] parent  A pointer to the parent CarriageWaySegmentTemplate.
    //!\return A shared pointer to the created LaneTemplate
    //----------------------------------------
    static Ptr create(CarriageWaySegmentTemplateWeakPtr parent) { return Ptr(new LaneTemplate(parent)); }

    //========================================
    //!\brief Constructor
    //!\param[in] id              The unique id of this Lane within the parent CarriageWaySegmentTemplate
    //!\param[in] laneId          The unique id of the Lane within the parent CarriageWayTemplate
    //!\param[in] type            The \ref LaneType of this LaneTemplate
    //!\param[in] parent          A pointer to the parent CarriageWaySegmentTemplate
    //!\param[in] nextLaneId      The id of the following Lane (0 if there is none)
    //!\param[in] previousLaneId  The id of the preceding Lane ( 0 if there is none)
    //!\param[in] leftLaneId      The id of the left neighboring Lane (0 if there is none)
    //!\param[in] rightLaneId     The id of the right neighboring Lane (0 if there is none)
    //----------------------------------------
    static Ptr create(const uint8_t& id,
                      const uint64_t laneId,
                      const LaneType& type,
                      CarriageWaySegmentTemplateWeakPtr parent,
                      const uint8_t& nextLaneId,
                      const uint8_t& previousLaneId,
                      const uint8_t& leftLaneId,
                      const uint8_t& rightLaneId)
    {
        return Ptr(new LaneTemplate(id, laneId, type, parent, nextLaneId, previousLaneId, leftLaneId, rightLaneId));
    }

    //========================================
    //! \brief Creates partly copy of the underlying object the other shared pointer is pointing to.
    //!
    //! \param[in] other  Shared pointer to copy from.
    //! \return A copy of the other object.
    //----------------------------------------
    static Ptr copy(const Ptr& other);

    //========================================
    // getter
    //========================================
public:
    //! Returns the unique id for the lane within the \ref CarriageWaySegmentTemplate.
    uint8_t getId() const { return m_id; }

    //! Returns the unique id of the lane within the parent \ref CarriageWayTemplate (0 if there is none).
    uint64_t getLaneId() const { return m_laneId; }

    //! Returns the type of the lane.
    LaneType getType() const { return m_type; }

    //! Returns the pointer to the parent \ref CarriageWaySegmentTemplate.
    const CarriageWaySegmentTemplatePtr getParent() const { return m_parent.lock(); }

    uint8_t getNextLaneId() const { return m_nextLaneId; }
    uint8_t getPrevLaneId() const { return m_prevLaneId; }
    uint8_t getLeftLaneId() const { return m_leftLaneId; }
    uint8_t getRightLaneId() const { return m_rightLaneId; }

    //! Return the pointer to the following lane (0 if there is none).
    const Ptr getNext() const { return m_nextLane.lock(); }

    //! Return the pointer to the preceding lane (0 if there is none).
    const Ptr getPrevious() const { return m_prevLane.lock(); }

    //! Return the pointer to the left neighboring lane (0 if there is none).
    const Ptr getLeft() const { return m_leftLane.lock(); }

    //! Return the pointer to the right neighboring lane (0 if there is none).
    const Ptr getRight() const { return m_rightLane.lock(); }

    //! Returns true if the Lane has a following lane, false otherwise.
    bool hasNext() const { return m_nextLane.expired() == false; }

    //! Returns true if the Lane has a preceding lane, false otherwise.
    bool hasPrevious() const { return m_prevLane.expired() == false; }

    //! Returns true if the Lane has a left neighboring lane, false otherwise.
    bool hasLeft() const { return m_leftLane.expired() == false; }

    //! Returns true if the Lane has a right neighboring lane, false otherwise.
    bool hasRight() const { return m_rightLane.expired() == false; }

    //! Returns a pointer to the map with all child LaneSegments.
    const LaneSegmentPtrMap& getLaneSegmentsMap() const { return m_laneSegmentsMap; }
    LaneSegmentPtrMap& getLaneSegmentsMap() { return m_laneSegmentsMap; }

    //========================================
    //!\brief Returns a bounding rectangle of the way which
    //!       is currently not the minimal one, but the one
    //!       aligned to the north vector.
    //----------------------------------------
    const BoundingRectangle& getBoundingBox() const { return m_boundingBox; }

    //========================================

    //========================================
    // setter
    //========================================
public:
    //! Sets the unique id for the lane within the \ref CarriageWaySegmentTemplate.
    void setId(const uint8_t id) { m_id = id; }

    //! Sets the unique id of the lane within the parent \ref CarriageWayTemplate.
    void setLaneId(const uint64_t& id) { m_laneId = id; }

    void setType(const LaneType type) { m_type = type; }

    //! Sets the pointer to the parent \ref CarriageWaySegmentTemplate.
    void setParent(const CarriageWaySegmentTemplateWeakPtr& parent) { m_parent = parent; }

    void setNextLaneId(const uint8_t nextId) { m_nextLaneId = nextId; }
    void setPrevLaneId(const uint8_t prevId) { m_prevLaneId = prevId; }
    void setLeftLaneId(const uint8_t leftId) { m_leftLaneId = leftId; }
    void setRightLaneId(const uint8_t rightId) { m_rightLaneId = rightId; }

    //! Sets the pointer to the next lane.
    void setNext(const Ptr& next)
    {
        m_nextLane   = next;
        m_nextLaneId = next ? next->getId() : 0;
    }

    //! Sets the pointer to the previous lane.
    void setPrevious(const Ptr& previous)
    {
        m_prevLane   = previous;
        m_prevLaneId = previous ? previous->getId() : 0;
    }

    //! Sets the pointer to the left lane.
    void setLeft(const Ptr& left)
    {
        m_leftLane   = left;
        m_leftLaneId = left ? left->getId() : 0;
    }

    //! Sets the pointer to the right lane.
    void setRight(const Ptr& right)
    {
        m_rightLane   = right;
        m_rightLaneId = right ? right->getId() : 0;
    }

    //! Sets all child LaneSegments to the given ones.
    void setSegments(const LaneSegmentPtrMap& segments) { m_laneSegmentsMap = segments; };

    //========================================
    // equality functions
    //========================================
public:
    static bool arePtrsInSameState(const typename LaneTemplate<TLaneSegment>::Ptr& lhs,
                                   const typename LaneTemplate<TLaneSegment>::Ptr& rhs);

    static bool arePtrsEqualNonRecursive(const typename LaneTemplate<TLaneSegment>::Ptr& lhs,
                                         const typename LaneTemplate<TLaneSegment>::Ptr& rhs);

    static bool areEqualRecursive(const LaneTemplate<TLaneSegment>& lhs, const LaneTemplate<TLaneSegment>& rhs);

    static bool areEqualNonRecursive(const LaneTemplate<TLaneSegment>& lhs, const LaneTemplate<TLaneSegment>& rhs);

public:
    //========================================

    //! Inserts a single LaneSegment to this lane.
    bool insert(LaneSegmentPtr segment)
    {
        // Inserts the segment to the map with its unique id and checks if another
        // segment with the same id already exists.
        return (m_laneSegmentsMap.insert(typename LaneSegmentPtrMap::value_type(segment->getId(), segment))).second;
    }

    void updateConnectionIds(const Ptr& reference, const bool override = true);

    uint64_t getNextFreeKey() const;

    void cleanIds();

    //========================================
    // Members
    //========================================
private:
    uint8_t m_id{0U};
    uint64_t m_laneId{0U};

    LaneType m_type{LaneType::Unclassified};
    CarriageWaySegmentTemplateWeakPtr m_parent{};

    uint8_t m_nextLaneId{0};
    uint8_t m_prevLaneId{0};
    uint8_t m_leftLaneId{0};
    uint8_t m_rightLaneId{0};

    WeakPtr m_nextLane{};
    WeakPtr m_prevLane{};
    WeakPtr m_leftLane{};
    WeakPtr m_rightLane{};
    LaneSegmentPtrMap m_laneSegmentsMap{};

    BoundingRectangle m_boundingBox{};
}; // LaneTemplate

//==============================================================================
// Implementation
//==============================================================================

template<typename TLaneSegment>
inline LaneTemplate<TLaneSegment>::LaneTemplate(const uint8_t& id,
                                                const uint64_t laneId,
                                                const LaneType& type,
                                                CarriageWaySegmentTemplateWeakPtr parent,
                                                const uint8_t& nextLaneId,
                                                const uint8_t& previousLaneId,
                                                const uint8_t& leftLaneId,
                                                const uint8_t& rightLaneId)
  : m_id(id),
    m_laneId(laneId),
    m_type(type),
    m_parent(parent),
    m_nextLaneId(nextLaneId),
    m_prevLaneId(previousLaneId),
    m_leftLaneId(leftLaneId),
    m_rightLaneId(rightLaneId)
{}

//==============================================================================

template<typename TLaneSegment>
inline typename LaneTemplate<TLaneSegment>::Ptr LaneTemplate<TLaneSegment>::copy(const Ptr& other)
{
    Ptr out(new LaneTemplate());

    out->setId(other->getId());
    out->setLaneId(other->getLaneId());
    out->setType(other->getType());
    out->setNextLaneId(other->getNextLaneId());
    out->setPrevLaneId(other->getPrevLaneId());
    out->setLeftLaneId(other->getLeftLaneId());
    out->setRightLaneId(other->getRightLaneId());

    return out;
}

//==============================================================================

template<typename TLaneSegment>
inline bool LaneTemplate<TLaneSegment>::arePtrsInSameState(const typename LaneTemplate<TLaneSegment>::Ptr& lhs,
                                                           const typename LaneTemplate<TLaneSegment>::Ptr& rhs)
{
    //	std::cerr << (void*)lhs.get() << "  " << (void*)rhs.get()
    //			<< "!lhs: " << (!lhs) << "    !rhs: " << (!rhs) << "    (lhs && rhs): " << (lhs && rhs) << "    "
    //			 << (((!lhs) && (!rhs)) || (lhs && rhs))
    //			 << std::endl;
    return ((!lhs) && (!rhs)) || (lhs && rhs);
}

//==============================================================================

template<typename TLaneSegment>
inline bool LaneTemplate<TLaneSegment>::arePtrsEqualNonRecursive(const typename LaneTemplate<TLaneSegment>::Ptr& lhs,
                                                                 const typename LaneTemplate<TLaneSegment>::Ptr& rhs)
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

//==============================================================================

template<typename TLaneSegment>
inline bool LaneTemplate<TLaneSegment>::areEqualRecursive(const LaneTemplate<TLaneSegment>& lhs,
                                                          const LaneTemplate<TLaneSegment>& rhs)
{
    const bool ok = (lhs.getType() == rhs.getType()) && (lhs.getBoundingBox() == rhs.getBoundingBox())
                    && (arePtrsEqualNonRecursive(lhs.getNext(), rhs.getNext()))
                    && (arePtrsEqualNonRecursive(lhs.getPrevious(), rhs.getPrevious()))
                    && (arePtrsEqualNonRecursive(lhs.getLeft(), rhs.getLeft()))
                    && (arePtrsEqualNonRecursive(lhs.getRight(), rhs.getRight()))
                    && (lhs.getLaneSegmentsMap().size() == rhs.getLaneSegmentsMap().size());

    //	const bool ok1 = (lhs.getType() == rhs.getType());
    //	const bool ok2 = (lhs.getBoundingBox() == rhs.getBoundingBox());
    //
    //	const bool ok3 = (arePtrsEqualNonRecursive(lhs.getNext(), rhs.getNext()));
    //	const bool ok4 = (arePtrsEqualNonRecursive(lhs.getPrevious(), rhs.getPrevious()));
    //	const bool ok5 = (arePtrsEqualNonRecursive(lhs.getLeft(), rhs.getLeft()));
    //	const bool ok6 = (arePtrsEqualNonRecursive(lhs.getRight(), rhs.getRight()));
    //
    //	const bool ok7 = (lhs.getLaneSegmentsMap().size() == rhs.getLaneSegmentsMap().size());
    //
    //	if (!ok1) { std::cerr << "!ok1" << std::endl; }
    //	if (!ok2) { std::cerr << "!ok2" << std::endl; }
    //	if (!ok3) { std::cerr << "!ok3" << std::endl; }
    //	if (!ok4) { std::cerr << "!ok4" << std::endl; }
    //	if (!ok5) { std::cerr << "!ok5" << std::endl; }
    //	if (!ok6) { std::cerr << "!ok6" << std::endl; }
    //	if (!ok7) { std::cerr << "!ok7" << std::endl; }

    if (!ok)
    {
        return false;
    }

    for (auto& lsme : lhs.getLaneSegmentsMap())
    {
        if (std::find_if(rhs.getLaneSegmentsMap().begin(),
                         rhs.getLaneSegmentsMap().end(),
                         [&lsme](const typename LaneTemplate<TLaneSegment>::LaneSegmentPtrMap::value_type& lsp) {
                             return TLaneSegment::areEqualRecursive(*lsme.second, *lsp.second);
                         })
            == rhs.getLaneSegmentsMap().end())
        {
            return false;
        }
    }

    return true;
}

//==============================================================================

template<typename TLaneSegment>
inline bool LaneTemplate<TLaneSegment>::areEqualNonRecursive(const LaneTemplate<TLaneSegment>& lhs,
                                                             const LaneTemplate<TLaneSegment>& rhs)
{
    const bool ok = (lhs.getType() == rhs.getType()) && (lhs.getBoundingBox() == rhs.getBoundingBox())
                    && (arePtrsInSameState(lhs.getNext(), rhs.getNext()))
                    && (arePtrsInSameState(lhs.getPrevious(), rhs.getPrevious()))
                    && (arePtrsInSameState(lhs.getLeft(), rhs.getLeft()))
                    && (arePtrsInSameState(lhs.getRight(), rhs.getRight()))
                    && (lhs.getLaneSegmentsMap().size() == rhs.getLaneSegmentsMap().size());

    //	const bool ok1 = (lhs.getType() == rhs.getType());
    //	const bool ok2 = (lhs.getBoundingBox() == rhs.getBoundingBox());
    //
    //	const bool ok3 = (arePtrsInSameState(lhs.getNext(), rhs.getNext()));
    //	const bool ok4 = (arePtrsInSameState(lhs.getPrevious(), rhs.getPrevious()));
    //	const bool ok5 = (arePtrsInSameState(lhs.getLeft(), rhs.getLeft()));
    //	const bool ok6 = (arePtrsInSameState(lhs.getRight(), rhs.getRight()));
    //
    //	const bool ok7 = (lhs.getLaneSegmentsMap().size() == rhs.getLaneSegmentsMap().size());
    //
    //	if (!ok1) { std::cerr << "!ok1" << std::endl; }
    //	if (!ok2) { std::cerr << "!ok2" << std::endl; }
    //	if (!ok3) { std::cerr << "!ok3" << std::endl; }
    //	if (!ok4) { std::cerr << "!ok4" << std::endl; }
    //	if (!ok5) { std::cerr << "!ok5" << std::endl; }
    //	if (!ok6) { std::cerr << "!ok6" << std::endl; }
    //	if (!ok7) { std::cerr << "!ok7" << std::endl; }

    if (!ok)
    {
        return false;
    }

    for (auto& lsme : lhs.getLaneSegmentsMap())
    {
        if (std::find_if(rhs.getLaneSegmentsMap().begin(),
                         rhs.getLaneSegmentsMap().end(),
                         [&lsme](const typename LaneTemplate<TLaneSegment>::LaneSegmentPtrMap::value_type& lsp) {
                             return TLaneSegment::areEqualRecursive(*lsme.second, *lsp.second);
                         })
            == rhs.getLaneSegmentsMap().end())
        {
            return false;
        }
    }

    return true;
}

//==============================================================================

template<typename TLaneSegment>
inline void LaneTemplate<TLaneSegment>::updateConnectionIds(const Ptr& reference, const bool override)
{
    if (reference)
    {
        if (m_leftLaneId == 0 || override)
        {
            m_leftLaneId = reference->getLeftLaneId();
        }
        if (m_rightLaneId == 0 || override)
        {
            m_rightLaneId = reference->getRightLaneId();
        }
        if (m_nextLaneId == 0 || override)
        {
            m_nextLaneId = reference->getNextLaneId();
        }
        if (m_prevLaneId == 0 || override)
        {
            m_prevLaneId = reference->getPrevLaneId();
        }
    }
}

//==============================================================================

template<typename TLaneSegment>
inline uint64_t LaneTemplate<TLaneSegment>::getNextFreeKey() const
{
    uint64_t maxKey = 0;
    for (auto& lsme : m_laneSegmentsMap)
    {
        if (lsme.first > maxKey)
        {
            maxKey = lsme.first;
        }
    }

    return ++maxKey;
}

//==============================================================================

template<typename TLaneSegment>
inline void LaneTemplate<TLaneSegment>::cleanIds()
{
    m_nextLaneId  = hasNext() ? getNext()->getId() : 0;
    m_prevLaneId  = hasPrevious() ? getPrevious()->getId() : 0;
    m_leftLaneId  = hasLeft() ? getLeft()->getId() : 0;
    m_rightLaneId = hasRight() ? getRight()->getId() : 0;

    for (auto& lsme : m_laneSegmentsMap)
    {
        lsme.second->cleanIds();
    }
}

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

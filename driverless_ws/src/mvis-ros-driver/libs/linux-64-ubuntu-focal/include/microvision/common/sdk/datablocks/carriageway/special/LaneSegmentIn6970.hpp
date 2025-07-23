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
//! \brief Segment of a lane
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/LaneIn6970.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/LaneSupportPointIn6970.hpp>
#include <microvision/common/sdk/LaneType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! A LaneSegmentIn6970 represents a \ref Lane segment within a parent Lane.
//! Each LaneSegment has a unique id within the parent \ref Lane
//!
//! A Lane Segment ends if the approximation error through straight bounding lines
//! reaches a certain value, or if a \ref CarriageWaySegmentIn6970 ends.
//!
//! The segmentation of a whole \ref CarriageWayIn6970 is as following:
//!
//!\ref CarriageWayIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref LaneIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6970
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create will return
//! a shared pointer to a new LaneSegment.
//!
//!\sa CarriageWayIn6970 \sa CarriageWaySegmentIn6970 \sa LaneIn6970
//------------------------------------------------------------------------------
class LaneSegmentIn6970 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;

public:
    // Shared/weak pointer to this class.
    using Ptr     = std::shared_ptr<LaneSegmentIn6970>;
    using WeakPtr = std::weak_ptr<LaneSegmentIn6970>;

    // Shared/weak pointer to parent classes.
    using LanePtr     = std::shared_ptr<LaneIn6970>;
    using LaneWeakPtr = std::weak_ptr<LaneIn6970>;

public: // static methods
    static Ptr copy(const Ptr& other);

private: // constructor
    //! Private default constructor called by \ref create \sa create.
    LaneSegmentIn6970() = default;

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6970(const LanePtr& parent);

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6970(const LaneSupportPointIn6970& laneStart);

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6970(const uint64_t& id,
                      const LaneMarkingType& markingLeft,
                      const LaneMarkingType& markingRight,
                      const LaneBoundaryType& boundaryLeft,
                      const LaneBoundaryType& boundaryRight,
                      const LaneSupportPointIn6970& laneStart,
                      const uint64_t& nextId,
                      const uint64_t& prevId,
                      const uint64_t& leftId,
                      const uint64_t& rightId,
                      const bool& nextInNewSeg,
                      const bool& prevInNewSeg,
                      LanePtr parent);

public: // destructor
    virtual ~LaneSegmentIn6970() {}

public:
    //========================================
    //!\brief Default constructor
    //!
    //! Initializes all elements to 0.
    //!\return A shared pointer to the created segment
    //----------------------------------------
    static Ptr create();

    //========================================
    //!\brief Constructor
    //!\param[in] parent  A pointer to the parent Lane.
    //!\return A shared pointer to the created segment.
    //----------------------------------------
    static Ptr create(LanePtr parent);

    //========================================
    //!\brief Constructor
    //!\param[in] laneStart  The start point of the LaneSegment.
    //----------------------------------------
    static Ptr create(const LaneSupportPointIn6970& laneStart);

    //========================================
    //!\brief Constructor
    //!\param[in] id            The unique id of the segment within the parent Lane
    //!\param[in] markingLeft   The type of the marking on the left side
    //!\param[in] markingRight  The type of the marking on the right side
    //!\param[in] borderLeft    The type of the border on the left side
    //!\param[in] borderRight   The type of the border on the right side
    //!\param[in] laneStart     The start point of the LaneSegment
    //!\param[in] nextId        The id of the next LaneSegment (0 if there is none)
    //!\param[in] prevId        The id of the preceding LaneSegment (0 if there is none)
    //!\param[in] leftId        The id of the left LaneSegment (0 if there is none)
    //!\param[in] rightId       The id of the right LaneSegment (0 if there is none)
    //!\param[in] nextInNewSeg  Information if next LaneSegment is in another CarriageWaySegment
    //!\param[in] prevInNewSeg  Information if preceding LaneSegment is in another CarriageWaySegment
    //!\param[in] parent        Pointer to the parent Lane
    //----------------------------------------
    static Ptr create(const uint64_t& id,
                      const LaneMarkingType& markingLeft,
                      const LaneMarkingType& markingRight,
                      const LaneBoundaryType& boundaryLeft,
                      const LaneBoundaryType& boundaryRight,
                      const LaneSupportPointIn6970& laneStart,
                      const uint64_t& nextId,
                      const uint64_t& prevId,
                      const uint64_t& leftId,
                      const uint64_t& rightId,
                      const bool& nextInNewSeg,
                      const bool& prevInNewSeg,
                      LanePtr parent);

public: // getter
    //! Returns the unique id of this LaneSegment within the parent Lane.
    uint64_t getId() const { return m_id; }

    //! Return the type of the left marking.
    LaneMarkingType getLeftMarkingType() const { return m_markingLeft; }

    //! Return the type of the right marking.
    LaneMarkingType getRightMarkingType() const { return m_markingRight; }

    //! Returns the type of the left border.
    LaneBoundaryType getLeftBoundaryType() const { return m_boundaryLeft; }

    //! Returns the type of the right border.
    LaneBoundaryType getRightBoundaryType() const { return m_boundaryRight; }

    //! Returns the start point of the segment.
    LaneSupportPointIn6970& getStartPoint() { return m_start; }
    const LaneSupportPointIn6970& getStartPoint() const { return m_start; }

    uint64_t getNextId() const { return m_nextId; }
    uint64_t getPrevId() const { return m_prevId; }
    uint64_t getLeftId() const { return m_leftId; }
    uint64_t getRightId() const { return m_rightId; }

    bool isNextInNewSeg() const { return m_nextInNewSeg; }
    bool isPrevInNewSeg() const { return m_prevInNewSeg; }

    //! Returns the length of the segment.
    float getLength() const { return m_length; }

    //========================================
    //!\brief Returns the width of the segment at the given
    //!       position. The position is relative to the start of
    //!       the segment and follows along the lane.
    //----------------------------------------
    float getWidth(const float position = 0.0F) const;

    //! Returns the offset from the center of the end point to the center of the start point.
    Vector2<float> getEndOffset() const { return m_endOffset; }

    //! Returns the offset from the left end bound to the center of the start point.
    Vector2<float> getEndOffsetLeft() const { return m_endOffsetLeft; }

    //! Returns the offset from the right end bound to the center of the start point.
    Vector2<float> getEndOffsetRight() const { return m_endOffsetRight; }

    //! Returns the offset from the left start bound to the center of the start point.
    Vector2<float> getStartOffsetLeft() const { return m_start.getLeftOffset(); }

    //! Returns the offset from the right start bound to the center of the start point.
    Vector2<float> getStartOffsetRight() const { return m_start.getRightOffset(); }

    //! Returns the pointer to the parent Lane (0 if not set).
    LanePtr getParent() const { return m_parent.lock(); }

    //! Returns a pointer to the following LaneSegment (0 if there is none).
    Ptr getNext() const { return m_nextSegment.lock(); }

    //! Returns a pointer to the preceding LaneSegment (0 if there is none).
    Ptr getPrevious() const { return m_prevSegment.lock(); }

    //! Returns a pointer to the left LaneSegment (0 if there is none).
    Ptr getLeft() const { return m_leftSegment.lock(); }

    //! Returns a pointer to the right LaneSegment (0 if there is none)
    Ptr getRight() const { return m_rightSegment.lock(); }

    //========================================
    //!\brief Returns a bounding rectangle of the way which is currently not the
    //!       minimal one, but the one aligned to the north vector.
    //----------------------------------------
    BoundingRectangle getBoundingBox() const { return m_boundingBox; }

    //! Returns the GPS position of the end point (center) of the segment.
    PositionWgs84 getEndGps();

public: // setter
    //! Sets the id of the lane segment.
    void setId(const uint64_t& id) { m_id = id; }

    void setLeftMarkingType(const LaneMarkingType type) { m_markingLeft = type; }
    void setRightMarkingType(const LaneMarkingType type) { m_markingRight = type; }

    void setLeftBorderType(const LaneBoundaryType type) { m_boundaryLeft = type; }
    void setRightBorderType(const LaneBoundaryType type) { m_boundaryRight = type; }

    void setStartPoint(const LaneSupportPointIn6970& point) { m_start = point; }

    void setNextId(const uint64_t& nextId) { m_nextId = nextId; }
    void setPrevId(const uint64_t& prevId) { m_prevId = prevId; }
    void setLeftId(const uint64_t& leftId) { m_leftId = leftId; }
    void setRightId(const uint64_t& rightId) { m_rightId = rightId; }

    void setNextInNewSeg(const bool inNewSeg) { m_nextInNewSeg = inNewSeg; }
    void setPrevInNewSeg(const bool inNewSeg) { m_prevInNewSeg = inNewSeg; }

    //! Sets the pointer to the parent Lane.
    void setParent(const LanePtr& parent) { m_parent = parent; }

    //! Sets the pointer to the next LaneSegment.
    void setNext(const Ptr& next);

    //! Sets the pointer to the previous LaneSegment.
    void setPrevious(const Ptr& previous);

    //! Sets the pointer to the left LaneSegment
    void setLeft(const Ptr& left);

    //! Sets the pointer to the right LaneSegment.
    void setRight(const Ptr& right);

    //! Returns true if there is a following segment, false otherwise.
    bool hasNext() const { return m_nextSegment.expired() == false; }

    //! Returns true if there is a preceding segment, false otherwise.
    bool hasPrevious() const { return m_prevSegment.expired() == false; }

    //! Returns true if there is a left segment, false otherwise.
    bool hasLeft() const { return m_leftSegment.expired() == false; }

    //! Returns true if there is a right segment, false otherwise.
    bool hasRight() const { return m_rightSegment.expired() == false; }

public:
    static bool arePtrsEqualNonRecursive(const LaneSegmentIn6970::Ptr& lhs, const LaneSegmentIn6970::Ptr& rhs);
    static bool areEqualRecursive(const LaneSegmentIn6970& lhs, const LaneSegmentIn6970& rhs);
    static bool areEqualNonRecursive(const LaneSegmentIn6970& lhs, const LaneSegmentIn6970& rhs);

public:
    void updateConnectionIds(const Ptr& reference, const bool override = true);
    void updateProperties();
    void cleanIds();

private:
    void calculateLength();
    void calculateWidth();
    void calculateBoundingBox();
    void calculateOffsets();

protected: // serialized members
    uint64_t m_id{0}; //!< The id of this LaneSegment (valid for the current Lane).
    LaneMarkingType m_markingLeft{LaneMarkingType::Unclassified};
    LaneMarkingType m_markingRight{LaneMarkingType::Unclassified};

    LaneBoundaryType m_boundaryLeft{LaneBoundaryType ::Unclassified};
    LaneBoundaryType m_boundaryRight{LaneBoundaryType::Unclassified};

    //BorderType m_borderLeft;
    //BorderType m_borderRight;

    LaneSupportPointIn6970 m_start{};

    uint64_t m_nextId{0}; //!< The id of the following LaneSegment.
    uint64_t m_prevId{0}; //!< The id of the previous LaneSegment.
    uint64_t m_leftId{0}; //!< The id of the left LaneSegment.
    uint64_t m_rightId{0}; //!< The id of the right LaneSegment.

    bool m_nextInNewSeg{false};
    bool m_prevInNewSeg{false};

private: // unserialized members
    float m_length{0.0F};
    float m_endWidth{0.0F};
    float m_startWidth{0.0F};

    Vector2<float> m_endOffset{};
    Vector2<float> m_endOffsetLeft{};
    Vector2<float> m_endOffsetRight{};

    LaneWeakPtr m_parent{};

    WeakPtr m_nextSegment{};
    WeakPtr m_prevSegment{};
    WeakPtr m_leftSegment{};
    WeakPtr m_rightSegment{};

    BoundingRectangle m_boundingBox{};
}; // LaneSegmentIn6970

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

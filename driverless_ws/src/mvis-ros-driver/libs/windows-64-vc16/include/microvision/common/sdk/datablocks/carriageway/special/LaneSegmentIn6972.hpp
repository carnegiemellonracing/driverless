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
//! \brief Segment of a lane
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/LaneIn6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/LaneSupportPointIn6972.hpp>
#include <microvision/common/sdk/LaneType.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! A LaneSegmentIn6972 represents a \ref Lane segment within a parent Lane.
//! Each LaneSegment has a unique id within the parent \ref Lane
//!
//! A Lane Segment ends if the approximation error through straight bounding lines
//! reaches a certain value, or if a \ref CarriageWaySegmentIn6972 ends.
//!
//! The segmentation of a whole \ref CarriageWayIn6972 is as following:
//!
//!\ref CarriageWayIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6972
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create will return
//! a shared pointer to a new LaneSegment.
//!
//!\sa CarriageWayIn6972 \sa CarriageWaySegmentIn6972 \sa LaneIn6972
//------------------------------------------------------------------------------
class LaneSegmentIn6972 final
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class microvision::common::sdk::Exporter;

public:
    // Shared/weak pointer to this class.
    using Ptr     = std::shared_ptr<LaneSegmentIn6972>;
    using WeakPtr = std::weak_ptr<LaneSegmentIn6972>;

    // Shared/weak pointer to parent classes.
    using LanePtr     = std::shared_ptr<LaneIn6972>;
    using LaneWeakPtr = std::weak_ptr<LaneIn6972>;

public: // static methods
    static Ptr copy(const Ptr& other);

private: // constructor
    //! Private default constructor called by \ref create \sa create.
    LaneSegmentIn6972();

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6972(const LanePtr& parent);

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6972(const LaneSupportPointIn6972& laneStart);

    //! Private constructor called by \ref create \sa create.
    LaneSegmentIn6972(const uint64_t& id,
                      const uint64_t nextId,
                      const uint64_t prevId,
                      const uint64_t leftId,
                      const uint64_t rightId,
                      const LaneMarkingType markingLeft,
                      const LaneMarkingType markingRight,
                      const LaneBoundaryType boundaryLeft,
                      const LaneBoundaryType boundaryRight,
                      const bool nextInNewSeg,
                      const bool prevInNewSeg,
                      const float markingWidthLeft,
                      const float markingWidthRigth,
                      const float medianDashLengthLeft,
                      const float medianDashLengthRight,
                      const float medianGapLengthLeft,
                      const float medianGapLengthRight,
                      const LaneSupportPointIn6972& laneStart,
                      LanePtr parent);

public: // destructor
    virtual ~LaneSegmentIn6972() {}

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
    static Ptr create(const LaneSupportPointIn6972& laneStart);

    //========================================
    //!\brief Create function
    //!\param[in] id                     The unique id of the segment within the parent Lane
    //!\param[in] nextId                 The id of the next LaneSegmentIn6970 (0 if there is none)
    //!\param[in] prevId                 The id of the preceding LaneSegmentIn6970 (0 if there is none)
    //!\param[in] leftId                 The id of the left LaneSegmentIn6970 (0 if there is none)
    //!\param[in] rightId                The id of the right LaneSegmentIn6970 (0 if there is none)
    //!\param[in] markingLeft            The type of the marking on the left side
    //!\param[in] markingRight           The type of the marking on the right side
    //!\param[in] boundaryLeft           The type of the border on the left side
    //!\param[in] boundaryRight          The type of the border on the right side
    //!\param[in] nextInNewSeg           Information if next LaneSegmentIn6970 is in another CarriageWaySegment
    //!\param[in] prevInNewSeg           Information if preceding LaneSegmentIn6970 is in another CarriageWaySegment
    //!\param[in] markingWidthLeft
    //!\param[in] markingWidthRight
    //!\param[in] medianDashLengthLeft
    //!\param[in] medianDashLengthRight
    //!\param[in] medianGapLengthLeft
    //!\param[in] medianGapLengthRight
    //!\param[in] laneStart              The start point of the LaneSegmentIn6970
    //!\param[in] parent                 Pointer to the parent Lane
    //----------------------------------------
    static Ptr create(const uint64_t id,
                      const uint64_t nextId,
                      const uint64_t prevId,
                      const uint64_t leftId,
                      const uint64_t rightId,
                      const LaneMarkingType markingLeft,
                      const LaneMarkingType markingRight,
                      const LaneBoundaryType boundaryLeft,
                      const LaneBoundaryType boundaryRight,
                      const bool nextInNewSeg,
                      const bool prevInNewSeg,
                      const float markingWidthLeft,
                      const float markingWidthRight,
                      const float medianDashLengthLeft,
                      const float medianDashLengthRight,
                      const float medianGapLengthLeft,
                      const float medianGapLengthRight,
                      const LaneSupportPointIn6972& laneStart,
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
    LaneSupportPointIn6972& getStartPoint() { return m_start; }
    const LaneSupportPointIn6972& getStartPoint() const { return m_start; }

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
    void setId(const uint64_t id) { m_id = id; }

    void setLeftMarkingType(const LaneMarkingType type) { m_markingLeft = type; }
    void setRightMarkingType(const LaneMarkingType type) { m_markingRight = type; }

    void setLeftBoundaryType(const LaneBoundaryType type) { m_boundaryLeft = type; }
    void setRightBoundaryType(const LaneBoundaryType type) { m_boundaryRight = type; }

    void setStartPoint(const LaneSupportPointIn6972& point) { m_start = point; }

    void setNextId(const uint64_t nextId) { m_nextId = nextId; }
    void setPrevId(const uint64_t prevId) { m_prevId = prevId; }
    void setLeftId(const uint64_t leftId) { m_leftId = leftId; }
    void setRightId(const uint64_t rightId) { m_rightId = rightId; }

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

    void setMarkingWidthLeft(const float markingWidthLeft) { m_markingWidthLeft = markingWidthLeft; }
    float getMarkingWidthLeft() const { return m_markingWidthLeft; }

    void setMarkingWidthRight(const float markingWidthRight) { m_markingWidthRight = markingWidthRight; }
    float getMarkingWidthRight() const { return m_markingWidthRight; }

    void setMedianDashLengthLeft(const float medianDashLengthLeft) { m_medianDashLengthLeft = medianDashLengthLeft; }
    float getMedianDashLengthLeft() const { return m_medianDashLengthLeft; }

    void setMedianDashLengthRight(const float medianDashLengthRight)
    {
        m_medianDashLengthRight = medianDashLengthRight;
    }
    float getMedianDashLengthRight() const { return m_medianDashLengthRight; }

    void setMedianGapLengthLeft(const float medianGapLengthLeft) { m_medianGapLengthLeft = medianGapLengthLeft; }
    float getMedianGapLengthLeft() const { return m_medianGapLengthLeft; }

    void setMedianGapLengthRight(const float medianGapLengthRight) { m_medianGapLengthRight = medianGapLengthRight; }
    float getMedianGapLengthRight() const { return m_medianGapLengthRight; }

    //! Returns true if there is a following segment, false otherwise.
    bool hasNext() const { return m_nextSegment.expired() == false; }

    //! Returns true if there is a preceding segment, false otherwise.
    bool hasPrevious() const { return m_prevSegment.expired() == false; }

    //! Returns true if there is a left segment, false otherwise.
    bool hasLeft() const { return m_leftSegment.expired() == false; }

    //! Returns true if there is a right segment, false otherwise.
    bool hasRight() const { return m_rightSegment.expired() == false; }

public:
    static bool arePtrsEqualNonRecursive(const LaneSegmentIn6972::Ptr& lhs, const LaneSegmentIn6972::Ptr& rhs);
    static bool areEqualRecursive(const LaneSegmentIn6972& lhs, const LaneSegmentIn6972& rhs);
    static bool areEqualNonRecursive(const LaneSegmentIn6972& lhs, const LaneSegmentIn6972& rhs);

public:
    void updateConnectionIds(const Ptr& reference, const bool override = true);
    void updateProperties();
    void cleanIds();

private:
    void init();
    void calculateLength();
    void calculateWidth();
    void calculateBoundingBox();
    void calculateOffsets();

private: // serialized members
    uint64_t m_id{0};

    uint64_t m_nextId{0};
    uint64_t m_prevId{0};
    uint64_t m_leftId{0};
    uint64_t m_rightId{0};

    LaneMarkingType m_markingLeft{LaneMarkingType::Unclassified};
    LaneMarkingType m_markingRight{LaneMarkingType::Unclassified};
    LaneBoundaryType m_boundaryLeft{LaneBoundaryType::Unclassified};
    LaneBoundaryType m_boundaryRight{LaneBoundaryType::Unclassified};

    bool m_nextInNewSeg{false};
    bool m_prevInNewSeg{false};

    float m_markingWidthLeft{0.0F};
    float m_markingWidthRight{0.0F};
    float m_medianDashLengthLeft{0.0F};
    float m_medianDashLengthRight{0.0F};
    float m_medianGapLengthLeft{0.0F};
    float m_medianGapLengthRight{0.0F};

    LaneSupportPointIn6972 m_start{};

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
}; // LaneSegmentIn6972

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

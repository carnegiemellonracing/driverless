//==============================================================================
//! \file
//!
//! \brief Base class for point cloud data containers.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 15, 2016
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/pointcloud/ReferencePlane.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

enum class PointType : uint16_t
{
    Point                     = 0x0000U,
    PointWithEpw              = 0x0001U,
    PointWithFlags            = 0x0002U,
    PointWithEpwAndFlags      = 0x0003U,
    PointWithEpwFlagsAndColor = 0x0004U
};

//==============================================================================

enum class PointKind : uint16_t
{
    Undefined      = 0x0000U,
    ScannedPoint   = 0x0001U,
    LanePoint      = 0x0002U,
    Curbstone      = 0x0003U,
    GuardRail      = 0x0004U,
    Roadmarking    = 0x0005U,
    OffroadMarking = 0x0006U
};

//==============================================================================

enum class PointFlag : uint32_t
{
    RoadMarking          = (1U << 0),
    Offroad              = (1U << 1),
    Curbstone            = (1U << 2),
    GuardRail            = (1U << 3),
    Unclassified         = (1U << 4),
    PointFlagReserved_05 = (1U << 5),
    PointFlagReserved_06 = (1U << 6),
    PointFlagReserved_07 = (1U << 7),
    PointFlagReserved_08 = (1U << 8),
    PointFlagReserved_09 = (1U << 9),
    PointFlagReserved_10 = (1U << 10),
    PointFlagReserved_11 = (1U << 11),
    PointFlagReserved_12 = (1U << 12),
    PointFlagReserved_13 = (1U << 13),
    PointFlagReserved_14 = (1U << 14),
    PointFlagReserved_15 = (1U << 15),
    PointFlagReserved_16 = (1U << 16),
    PointFlagReserved_17 = (1U << 17),
    PointFlagReserved_18 = (1U << 18),
    PointFlagReserved_19 = (1U << 19),
    PointFlagReserved_20 = (1U << 20),
    PointFlagReserved_21 = (1U << 21),
    PointFlagReserved_22 = (1U << 22),
    PointFlagReserved_23 = (1U << 23),
    PointFlagReserved_24 = (1U << 24),
    PointFlagReserved_25 = (1U << 25),
    PointFlagReserved_26 = (1U << 26),
    PointFlagReserved_27 = (1U << 27),
    PointFlagReserved_28 = (1U << 28),
    PointFlagReserved_29 = (1U << 29),
    PointFlagReserved_30 = (1U << 30),
    PointFlagReserved_31 = (1U << 31)
}; // PointFlag

//==============================================================================

extern bool pointHasEpw(const PointType type);
extern bool pointHasFlags(const PointType type);
extern bool pointHasColor(const PointType type);

//==============================================================================

class PointCloudBase
{
public:
    PointCloudBase();
    virtual ~PointCloudBase() {}

public:
    bool operator==(const PointCloudBase& other) const;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    static std::string kindToString(const PointKind kind);
    static PointKind stringToKind(const std::string& kind);

public:
    virtual PointKind getKind() const { return m_kind; }
    virtual PointType getType() const { return m_type; }

    virtual void setKind(const PointKind kind) { m_kind = kind; }
    virtual void setType(const PointType type) { m_type = type; }

    bool hasEpw() const { return sdk::pointHasEpw(m_type); }
    bool hasFlags() const { return sdk::pointHasFlags(m_type); }
    bool hasColor() const { return sdk::pointHasColor(m_type); }

    virtual ReferencePlane& referencePlane() { return m_refPlane; }
    virtual const ReferencePlane& getReferencePlane() const { return m_refPlane; }
    virtual void setReferencePlane(const ReferencePlane& plane) { m_refPlane = plane; }

protected:
    PointKind m_kind;
    PointType m_type;
    ReferencePlane m_refPlane;
}; // PointCloudBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

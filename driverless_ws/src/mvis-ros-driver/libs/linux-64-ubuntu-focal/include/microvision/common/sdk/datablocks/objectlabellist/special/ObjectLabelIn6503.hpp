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
//! \date Apr 28, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/special/MeasurementSeriesIn6503.hpp>
#include <microvision/common/sdk/ObjectBasic.hpp>
#include <microvision/common/sdk/Rectangle.hpp>

#include <array>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

using Uoid = uint64_t;

//==============================================================================

class ObjectLabelIn6503 final
{
public:
    static const uint32_t nbOfReserved = 3;

public: // type declaration
    using ReservedArray = std::array<uint16_t, nbOfReserved>;

public:
    ObjectLabelIn6503()          = default;
    virtual ~ObjectLabelIn6503() = default;

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public: // getter
    const Rectangle<int16_t>& getObjBox() const { return m_objBox; }
    Rectangle<int16_t>& getObjBox() { return m_objBox; }
    uint8_t getWeight() const { return m_weight; }
    uint8_t getClassification() const { return m_classification; }
    uint32_t getId() const { return m_id; }
    Uoid getTrackingId() const { return m_trackingId; }
    float getDisplayMagnificationFactor() const { return m_displayMagnificationFactor; }
    uint16_t getObjectFlags() const { return m_objectFlags; }
    uint16_t getIsInOnKeyframe() const { return m_isInOnKeyframe; }
    uint16_t getReserved(const uint32_t idx) const { return m_reserved.at(idx); }
    const MeasurementSeriesIn6503& getUserData() const { return m_userData; }
    MeasurementSeriesIn6503& getUserData() { return m_userData; }

public: // setter
    // use getUserData
    void setWeight(uint8_t newWeight) { m_weight = newWeight; }
    void setClassification(const uint8_t newClassification) { m_classification = newClassification; }
    void setId(const uint32_t newId) { m_id = newId; }
    void setTrackingId(const Uoid newTrackingId) { m_trackingId = newTrackingId; }
    void setDisplayMagnificationFactor(const float newDmf) { m_displayMagnificationFactor = newDmf; }
    void setObjectFlags(const uint16_t newObjectFlags) { m_objectFlags = newObjectFlags; }
    void setIsInOnKeyframe(const uint16_t newIsInOnKeyframe) { m_isInOnKeyframe = newIsInOnKeyframe; }
    void setReserved(const uint32_t idx, const uint16_t newReserved) { m_reserved.at(idx) = newReserved; }
    // use getUserData

protected:
    Rectangle<int16_t> m_objBox{};
    uint8_t m_weight{0};
    uint8_t m_classification{static_cast<uint8_t>(ObjectClass::Unclassified)};
    uint32_t m_id{0};
    Uoid m_trackingId{0};
    float m_displayMagnificationFactor{1.0F};
    uint16_t m_objectFlags{0};
    uint16_t m_isInOnKeyframe{0};
    ReservedArray m_reserved{};
    MeasurementSeriesIn6503 m_userData{};
}; // ObjectLabel6503

//==============================================================================

bool operator==(const ObjectLabelIn6503& lhs, const ObjectLabelIn6503& rhs);
inline bool operator!=(const ObjectLabelIn6503& lhs, const ObjectLabelIn6503& rhs) { return !(lhs == rhs); }

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

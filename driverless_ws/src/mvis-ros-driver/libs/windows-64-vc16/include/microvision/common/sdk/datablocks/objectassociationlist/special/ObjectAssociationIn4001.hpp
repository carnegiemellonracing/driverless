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
//! \date Apr 26, 2016
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/NtpTime.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectAssociationIn4001 final
{
public:
    ObjectAssociationIn4001()          = default;
    virtual ~ObjectAssociationIn4001() = default;

    //==============================================================================

public: // getter
    uint32_t getReferenceObjId() const { return m_refObjId; }
    uint32_t getDutObjId() const { return m_dutObjId; }
    NtpTime getTimestampFirst() const { return m_timestampFirst; }
    NtpTime getTimestampLast() const { return m_timestampLast; }
    float getCertainty() const { return m_certainty; }

public: // setter
    void setRefObjId(const uint32_t newReferenceObjectId) { m_refObjId = newReferenceObjectId; }
    void setDutObjId(const uint32_t newDutObjectId) { m_dutObjId = newDutObjectId; }
    void setTimestampFirst(const NtpTime newTimestampFirst) { m_timestampFirst = newTimestampFirst; }
    void setTimestampLast(const NtpTime newTimestampLast) { m_timestampLast = newTimestampLast; }
    void setCertainty(const float newCertainty) { m_certainty = newCertainty; }

public:
    static std::streamsize getSerializedSize_static() { return 28; }

public:
    virtual std::streamsize getSerializedSize() const { return getSerializedSize_static(); }
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

protected:
    uint32_t m_refObjId{0}; //!< Object Id of the reference object involved in association.
    uint32_t m_dutObjId{0}; //!<Object Id of the DuT object involved in association.
    NtpTime m_timestampFirst{}; //!<Timestamp when the association begins.
    NtpTime m_timestampLast{}; //!<  Timestamp when the association ends .
    float m_certainty{0.0F}; //!< Certainty of the association, [0 .. 1].
}; // ObjectAssociation4001Entry

//==============================================================================

inline bool operator==(const ObjectAssociationIn4001& lhs, const ObjectAssociationIn4001& rhs)
{
    return lhs.getReferenceObjId() == rhs.getReferenceObjId() && lhs.getDutObjId() == rhs.getDutObjId()
           && lhs.getTimestampFirst() == rhs.getTimestampFirst() && lhs.getTimestampLast() == rhs.getTimestampLast()
           && fuzzyFloatEqualT<6>(lhs.getCertainty(), rhs.getCertainty());
}

//==============================================================================

inline bool operator!=(const ObjectAssociationIn4001& lhs, const ObjectAssociationIn4001& rhs) { return !(lhs == rhs); }

//==============================================================================

} // namespace sdk
} // namespace common
} // namespace microvision

//==============================================================================

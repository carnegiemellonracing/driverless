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
//! \date Jan 17, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2225.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU object data:
//! ObjectList has now a timestamp that should have the midTimestamp of the corresponding scan
//!
//! Object data available from FUSION SYSTEM and AppBase2 (ECU).
//! Each data block starts with a header followed by the object list.
//! Each object has a list of contour points.
//! The Sigma values are calculated using a Kalman filter by taking into account the object age.
//! All positions and angles are given in the vehicle / reference coordinate system.
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2225 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2225>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2225"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2225();
    virtual ~ObjectList2225();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    NtpTime getTimestamp() const { return m_timestamp; }

    uint16_t getNbOfObjects() const { return static_cast<uint16_t>(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setTimestamp(const NtpTime newTimeStamp) { m_timestamp = newTimeStamp; }
    void setObjects(const ObjectVector& newObjects) { m_objects = newObjects; }

protected:
    NtpTime m_timestamp{0}; //!< Mid-scan timestamp of the corresponding scan.
    ObjectVector m_objects{}; //!< Vector of objects.
}; // ObjectList2225

//==============================================================================

//==============================================================================

bool operator==(const ObjectList2225& lhs, const ObjectList2225& rhs);
bool operator!=(const ObjectList2225& lhs, const ObjectList2225& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2290.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Same as DataTypeObjectList (0x2280) but holds only reference objects.
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2290 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2290>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2290"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2290();
    virtual ~ObjectList2290();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    NtpTime getTimestamp() const { return m_timestamp; }

    uint16_t getNbOfObjects() const { return static_cast<uint16_t>(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setTimestamp(const NtpTime& timestamp) { m_timestamp = timestamp; }
    void setObjects(const ObjectVector& objects) { m_objects = objects; }

protected:
    NtpTime
        m_timestamp; //!< The absolute timestamp when the scanner mirror crossed the middle of the corresponding scan.
    // number of objects uint16_t
    ObjectVector m_objects; //! Vector of objects.
}; // ObjectList2290

//==============================================================================

bool operator==(const ObjectList2290& lhs, const ObjectList2290& rhs);
bool operator!=(const ObjectList2290& lhs, const ObjectList2290& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2280.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU object data:
//! List of objects with extended tracking information (generic)
//!
//! Object data available from FUSION SYSTEM and MVIS ECU connected with laser scanners.
//! Each data block starts with the IdcDataHeader followed by the object list.
//! For each object list this header is preceded.
//!
//! Each object has a list of contour points. Subtypes are described below.
//
//! All positions and angles are given in the vehicle / reference coordinate system.
//
//! In general, positions, lengths, distances and sizes are coded in meters. In general, angles are coded in radians.
//!
//! Note: depending on the configuration of the fusion system, the ECU can provide object data of type 0x2281.
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2280 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2280>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2280"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2280();
    virtual ~ObjectList2280();

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
    ObjectVector m_objects; //!< Vector of objects.
}; // ObjectList2280

//==============================================================================

bool operator==(const ObjectList2280& lhs, const ObjectList2280& rhs);
bool operator!=(const ObjectList2280& lhs, const ObjectList2280& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

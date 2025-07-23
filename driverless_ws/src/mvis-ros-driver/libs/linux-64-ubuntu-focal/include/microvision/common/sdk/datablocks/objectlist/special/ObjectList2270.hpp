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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2270.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of objects with extended tracking information in the format of the LUX4/Scala145 firmware
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2270 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2270>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2270"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    ObjectList2270();
    ObjectList2270(const NtpTime scanStartTimeStamp);
    virtual ~ObjectList2270();

public:
    const ObjectIn2270* findObject(const uint16_t id) const;
    void addObject(const ObjectIn2270& newObj);

public: // getter
    NtpTime getScanStartTimestamp() const { return m_scanStartTimestamp; }
    uint16_t getScanNumber() const { return m_scanNumber; }
    uint16_t getNbOfObjects() const { return static_cast<uint16_t>(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setScanStartTimeStamp(const NtpTime newScanStartTimeStamp) { m_scanStartTimestamp = newScanStartTimeStamp; }
    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }

protected:
    NtpTime m_scanStartTimestamp; //!< Timestamp of the first scan these objects are updated with.
    uint16_t m_scanNumber{0}; //!< The scan number of the first scan these objects are updated with.
    ObjectVector m_objects{}; //!< Vector of objects in this list.
}; // ObjectList2270

//==============================================================================

//==============================================================================

bool operator==(const ObjectList2270& lhs, const ObjectList2270& rhs);
bool operator!=(const ObjectList2270& lhs, const ObjectList2270& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

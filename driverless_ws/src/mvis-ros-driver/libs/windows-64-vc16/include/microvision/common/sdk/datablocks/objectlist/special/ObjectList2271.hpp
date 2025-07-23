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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2271.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of objects for scala. With unfiltered and filtered object information.
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2271 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2271>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2271"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2271();
    ObjectList2271(const NtpTime scanStartTimeStamp);

    virtual ~ObjectList2271();

public: // getter
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    NtpTime getScanStartTimestamp() const { return m_scanStartTimestamp; }
    uint16_t getScanNumber() const { return m_scanNumber; }
    uint8_t getObjectListId() const { return m_objectListId; }
    uint8_t getDeviceId() const { return m_deviceId; }
    uint16_t getDeviceInterfaceVersion() const { return m_deviceInterfaceVersion; }

    uint32_t getReserved() const { return m_reserved; }

    uint16_t getNbOfObjects() const { return uint16_t(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setScanStartTimestamp(const NtpTime& newNtpTime) { m_scanStartTimestamp = newNtpTime; }
    void setScanNumber(const uint16_t newScanNumber) { m_scanNumber = newScanNumber; }
    void setObjectListId(const uint8_t newObjectListId) { m_objectListId = newObjectListId; }
    void setDeviceId(const uint8_t newDeviceId) { m_deviceId = newDeviceId; }
    //========================================
    //! \brief Set the object list id. (14 bit value).
    //! \attention Only the lower 14 bits are used, i.e. the
    //!            device interface version is between 0 an 16383.
    //----------------------------------------
    void setDeviceInterfaceVersion(const uint16_t newDeviceInterfaceVersion)
    {
        assert((newDeviceInterfaceVersion & 0x3FFF) == newDeviceInterfaceVersion);
        m_deviceInterfaceVersion = (newDeviceInterfaceVersion & 0x3FFF);
    }

    void setReserved(const uint32_t newReserved) { m_reserved = newReserved; }
    void setObjects(const ObjectVector& objects) { m_objects = objects; }

protected:
    NtpTime m_scanStartTimestamp; //!< The timestamp of the first scan these objects are updated with.
    uint16_t m_scanNumber{0}; //!< The scan number of the first scan these objects are updated with.
    uint8_t m_objectListId{0}; //!< The id of this object list.
    uint8_t m_deviceId{0}; //!< The id of the device creating this object list.
    uint16_t m_deviceInterfaceVersion{0}; //!< The version number of the device.
    uint32_t m_reserved{0}; //!< Reserved.
    // uint16 number of objects
    ObjectVector m_objects{}; //!< Vector of objects in this list.
}; // ObjectList2271

//==============================================================================

bool operator==(const ObjectList2271& lhs, const ObjectList2271& rhs);
bool operator!=(const ObjectList2271& lhs, const ObjectList2271& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

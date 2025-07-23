//==============================================================================
//! \file
//!
//! \brief Generalized container class for object lists
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 9, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/objectlist/Object.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief List of objects
//!
//! Special data types:
//! \ref microvision::common::sdk::ObjectList2225
//! \ref microvision::common::sdk::ObjectList2270
//! \ref microvision::common::sdk::ObjectList2271
//! \ref microvision::common::sdk::ObjectList2280
//! \ref microvision::common::sdk::ObjectList2281
//! \ref microvision::common::sdk::ObjectList2290
//! \ref microvision::common::sdk::ObjectList2291
//------------------------------------------------------------------------------
class ObjectList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;
    friend class ObjectListImporter2280_2290;
    friend class ObjectListImporter2281_2291;
    friend class ObjectListExporter2281_2291;

public:
    // Coordinate system that the objects are defined relative to (i.e. position of origin on the ego vehicle).
    enum class CoordinateSystem : uint8_t
    {
        MiddleRearAxisISO70000 = 0x00, // Vehicle Coordinate System ISO70000 (default)
        MiddleFrontAxis        = 0x01, // Conti's reference coordinate system (Middle Front Axis, z=0).
        Undefined              = 0x0F // Invalid/unspecified coordinate system.
    };

    enum class DeviceType : uint8_t
    {
        Undefined    = 0,
        Laserscanner = 1,
        Radar        = 50,
        Camera       = 60,
        Reference    = 90,
        DuT          = 120
    };

    // Flags
    static const uint8_t coordinateSystemMask{0x0F};
    static const uint8_t referenceObjectsFlag{0x10};

public:
    using ObjectVector = std::vector<Object>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.objectlist"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList()          = default;
    virtual ~ObjectList() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    NtpTime getTimestamp() const { return m_timestamp; }
    uint8_t getObjectListId() const { return m_objectListId; }
    uint8_t getFlags() const { return m_flags; }
    CoordinateSystem getCoordinateSystem() const
    {
        return static_cast<CoordinateSystem>(m_flags & coordinateSystemMask);
    }
    DeviceType getDeviceType() const { return m_deviceType; }
    uint16_t getDeviceInterfaceVersion() const { return m_deviceInterfaceVersion; }
    bool isReferenceObjectsOnly() const { return (m_flags & referenceObjectsFlag) != 0; }

    uint16_t getNbOfObjects() const { return static_cast<uint16_t>(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setTimestamp(const NtpTime& newTimestamp) { m_timestamp = newTimestamp; }
    void setObjectListId(uint8_t newObjectListId) { m_objectListId = newObjectListId; }
    void setFlags(const uint8_t flags) { m_flags = flags; }
    void setCoordinateSystem(CoordinateSystem coordinateSystem)
    {
        m_flags = static_cast<uint8_t>((m_flags & (~coordinateSystemMask)) | static_cast<uint8_t>(coordinateSystem));
    }
    void setDeviceType(DeviceType newDeviceType) { m_deviceType = newDeviceType; }
    void setDeviceInterfaceVersion(uint16_t newDeviceInterfaceVersion)
    {
        m_deviceInterfaceVersion = newDeviceInterfaceVersion;
    }
    void setReferenceObjectsOnly(bool referenceObjectsOnly)
    {
        referenceObjectsOnly ? m_flags |= referenceObjectsFlag : m_flags &= static_cast<uint8_t>(~referenceObjectsFlag);
    }
    void setObjects(const ObjectVector& objects) { m_objects = objects; }

protected:
    NtpTime m_timestamp{0};
    uint8_t m_objectListId{0xFF};
    uint8_t m_flags{0xFF};
    DeviceType m_deviceType{DeviceType::Undefined};
    uint16_t m_deviceInterfaceVersion{0xFFFF};
    ObjectVector m_objects;
}; // ObjectList

//==============================================================================

bool operator==(const ObjectList& lhs, const ObjectList& rhs);
bool operator!=(const ObjectList& lhs, const ObjectList& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

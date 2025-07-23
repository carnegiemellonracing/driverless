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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2281.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief FUSION SYSTEM/ECU object data:
//! List of objects with extended tracking information an dynamic Type (generic)
//!
//! Object data are available from the FUSION SYSTEM and the MVIS ECU connected with laser scanners.
//! Each data block starts with the IdcDataHeader followed by the object list.
//! The IdcDataHeader precedes each object list. The IdcDataHeader is described in Section 2.4 idc data Header.
//!
//! All positions and angles are given in the vehicle / reference coordinate system.
//!
//! In general, positions, lengths, distances and sizes are coded in meters. In general, angles are coded in radians.
//!
//! General data type: \ref microvision::common::sdk::ObjectList
//------------------------------------------------------------------------------
class ObjectList2281 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    using ObjectVector = std::vector<ObjectIn2281>;

public:
    static const uint8_t flagBits_MiddleRearAxisISO70000 = 0x00U;
    static const uint8_t flagBits_MiddleFrontAxis        = 0x01U;
    static const uint8_t flagBits_CoordianteSystemMask   = 0x0FU;

    static const uint8_t flagBits_isRefObjList = 0x10U;

    static const uint8_t flagBits_reservedMask = 0xE0U;

    //! Device Type that created this object list
    enum class DeviceType : uint8_t
    {
        Undefined    = 0,
        Laserscanner = 1,
        Radar        = 50,
        Camera       = 60,
        Reference    = 90,
        DuT          = 120
    };

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectlist2281"};

    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectList2281();
    ~ObjectList2281() override;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public: // getter
    NtpTime getTimestamp() const { return m_timestamp; }
    uint8_t getObjectListId() const { return m_objectListId; }
    DeviceType getDeviceType() const { return m_deviceType; }
    uint16_t getDeviceInterfaceVersion() const { return m_deviceInterfaceVersion; }

    bool coordSystemIsMiddleRearAxis() const
    {
        return (m_flags & flagBits_CoordianteSystemMask) == flagBits_MiddleRearAxisISO70000;
    }
    bool coordSystemIsMiddleFrontAxis() const
    {
        return (m_flags & flagBits_CoordianteSystemMask) == flagBits_MiddleFrontAxis;
    }
    bool isRefObjList() const { return (m_flags & flagBits_isRefObjList) == flagBits_isRefObjList; }

    uint8_t getFlags() const { return m_flags; }
    uint8_t getReserved1() const { return m_reserved1; }

    uint16_t getNbOfObjects() const { return uint16_t(m_objects.size()); }
    const ObjectVector& getObjects() const { return m_objects; }
    ObjectVector& getObjects() { return m_objects; }

public: // setter
    void setTimestamp(const NtpTime& newTimestamp) { m_timestamp = newTimestamp; }
    void setObjectListId(const uint8_t newObjectListId) { m_objectListId = newObjectListId; }
    void setDeviceType(const DeviceType newDeviceType) { m_deviceType = newDeviceType; }
    void setDeviceInterfaceVerstion(const uint16_t newDeviceInterfaceVersion)
    {
        m_deviceInterfaceVersion = newDeviceInterfaceVersion;
    }

    void setCoordSystemToMiddleRearAxis()
    {
        m_flags = uint8_t(m_flags & ~flagBits_CoordianteSystemMask) | flagBits_MiddleRearAxisISO70000;
    }
    void setCoordSystemToISO70000() { this->setCoordSystemToMiddleRearAxis(); }
    void setCoordSystemToMiddleFrontAxis()
    {
        m_flags = uint8_t(uint8_t(m_flags & ~flagBits_CoordianteSystemMask) | flagBits_MiddleFrontAxis);
    }

    void setToBeRefObjList(const bool toBe)
    {
        if (toBe)
            m_flags = uint8_t(m_flags | flagBits_isRefObjList);
        else
            m_flags = uint8_t(m_flags & ~flagBits_isRefObjList);
    }

    void setFlags(const uint8_t newFlags) { m_flags = newFlags; }
    void setReserved1(const uint8_t newReserved1) { m_reserved1 = newReserved1; }

    void setObjects(const ObjectVector& newObjects) { m_objects = newObjects; }

protected:
    //! The absolute timestamp when the scanner mirror crossed the middle of the corresponding scan.
    NtpTime m_timestamp{0};

    //! The unique object list identifier to match object list with its source of computation.
    uint8_t m_objectListId{0};

    DeviceType m_deviceType{DeviceType::Undefined}; //!<The device type that created this object list.

    //! The device interface version of software that creates this object list.
    //! Only 14 Bits can be used (0-16383).
    uint16_t m_deviceInterfaceVersion{0};

    uint8_t m_flags{0}; //!< The object list flags.
    uint8_t m_reserved1{0}; //!< Reserved.
    //nbOfObjects
    ObjectVector m_objects; //!< The vector of objects.
}; // ObjectList2281

//==============================================================================

template<>
void readBE<ObjectList2281::DeviceType>(std::istream& is, ObjectList2281::DeviceType& dt);

template<>
void writeBE<ObjectList2281::DeviceType>(std::ostream& os, const ObjectList2281::DeviceType& dt);

//==============================================================================

bool operator==(const ObjectList2281& lhs, const ObjectList2281& rhs);
bool operator!=(const ObjectList2281& lhs, const ObjectList2281& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

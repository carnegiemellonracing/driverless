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
//! \date Jan 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/ObjectAssociation.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationList4001.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>
#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class ObjectAssociationList final : public DataContainerBase
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

    friend bool operator==(const ObjectAssociationList& lhs, const ObjectAssociationList& rhs);

public:
    static constexpr uint32_t nbOfReserved{ObjectAssociationList4001::nbOfReserved};
    static constexpr uint16_t devInterfaceVersionMask{ObjectAssociationList4001::devInterfaceVersionMask};
    static constexpr uint8_t objListIdMask{ObjectAssociationList4001::objListIdMask};

public: // type declaration
    using ObjAssocVector = std::vector<ObjectAssociation>;
    using ReservedArray  = std::array<char, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.generalcontainer.objectassociationlist"};
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectAssociationList() : DataContainerBase() {}
    virtual ~ObjectAssociationList() = default;

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint8_t getRefObjListId() const { return m_delegate.getRefObjListId(); }
    uint8_t getRefDevType() const { return m_delegate.getRefDevType(); }
    uint16_t getRefDevInterfaceVersion() const { return m_delegate.getRefDevInterfaceVersion(); }
    uint8_t getDutObjListId() const { return m_delegate.getDutObjListId(); }
    uint8_t getDutDevType() const { return m_delegate.getDutDevType(); }
    uint16_t getDutDevInterfaceVersion() const { return m_delegate.getDutDevInterfaceVersion(); }

    const ObjAssocVector& getObjectAssociations() const { return m_delegate.getObjectAssociations(); }
    ObjAssocVector& getObjectAssociations() { return m_delegate.getObjectAssociations(); }

    const ReservedArray& getReserved() const { return m_delegate.getReserved(); }
    char getReserved(const uint8_t r) const { return m_delegate.getReserved(r); }

public:
    void setRefObjListId(const uint8_t newRefObjListId) { m_delegate.setRefObjListId(newRefObjListId); }
    void setRefDevType(const uint8_t newRefDevType) { m_delegate.setRefDevType(newRefDevType); }
    void setRefDevInterfaceVersion(const uint16_t newRefDevInterfaceVersion)
    {
        m_delegate.setRefDevInterfaceVersion(newRefDevInterfaceVersion);
    }
    void setDutObjListId(const uint8_t newDutObjListId) { m_delegate.setDutObjListId(newDutObjListId); }
    void setDutDevType(const uint8_t newDutDevType) { m_delegate.setDutDevType(newDutDevType); }
    void setDutDevInterfaceVersion(const uint16_t newDutDevInterfaceVersion)
    {
        m_delegate.setDutDevInterfaceVersion(newDutDevInterfaceVersion);
    }

    //========================================
    //! \brief Setter for reserved values.
    //! \attention Not part of the "public" interface For testing purpose only!
    //----------------------------------------
    void setReserved(const uint8_t r, const char newReservedValue) { m_delegate.setReserved(r, newReservedValue); }

private:
    ObjectAssociationList4001 m_delegate;
}; // ObjectAssociationList

//==============================================================================

inline bool operator==(const ObjectAssociationList& lhs, const ObjectAssociationList& rhs)
{
    return lhs.m_delegate == rhs.m_delegate;
}

inline bool operator!=(const ObjectAssociationList& lhs, const ObjectAssociationList& rhs) { return !(lhs == rhs); }

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

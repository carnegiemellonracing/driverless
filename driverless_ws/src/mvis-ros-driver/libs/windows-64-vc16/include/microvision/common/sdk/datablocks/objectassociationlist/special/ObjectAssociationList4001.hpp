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
//! \date Mar 20, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/DataContainerBase.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationIn4001.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Object Association
//!
//! General data type: \ref microvision::common::sdk::ObjectAssociationList
//------------------------------------------------------------------------------
class ObjectAssociationList4001 final : public SpecializedDataContainer
{
    template<class ContainerType, DataTypeId::DataType id>
    friend class Importer;
    template<class ContainerType, DataTypeId::DataType id>
    friend class Exporter;

public:
    static const uint32_t nbOfReserved = 8;
    static const uint16_t devInterfaceVersionMask{0x3FFF};
    static const uint8_t objListIdMask{0xFF};

public: // type declaration
    using ObjAssocVector = std::vector<ObjectAssociationIn4001>;
    using ReservedArray  = std::array<char, nbOfReserved>;

public:
    constexpr static const char* const containerType{"sdk.specialcontainer.objectassociationlist4001"};
    static constexpr uint64_t getClassIdHashStatic() { return hash(containerType); }

public:
    ObjectAssociationList4001();
    virtual ~ObjectAssociationList4001();

public:
    uint64_t getClassIdHash() const override { return getClassIdHashStatic(); }

public:
    uint8_t getRefObjListId() const { return m_refObjListId; }
    uint8_t getRefDevType() const { return m_refDevType; }
    uint16_t getRefDevInterfaceVersion() const { return m_refDevInterfaceVersion; }
    uint8_t getDutObjListId() const { return m_dutObjListId; }
    uint8_t getDutDevType() const { return m_dutDevType; }
    uint16_t getDutDevInterfaceVersion() const { return m_dutDevInterfaceVersion; }

    const ObjAssocVector& getObjectAssociations() const { return m_objAssocs; }
    ObjAssocVector& getObjectAssociations() { return m_objAssocs; }

    const ReservedArray& getReserved() const { return m_reserved; }

    char getReserved(const uint8_t r) const
    {
        if (r < nbOfReserved)
        {
            return m_reserved.at(r);
        }
        else
        {
            return static_cast<char>(-1); //changed: was char(0xFF) == -1 (def. value: maybe not set)
        }
    }

public:
    void setRefObjListId(const uint8_t newRefObjListId) { m_refObjListId = newRefObjListId; }
    void setRefDevType(const uint8_t newRefDevType) { m_refDevType = newRefDevType; }
    void setRefDevInterfaceVersion(const uint16_t newRefDevInterfaceVersion)
    {
        m_refDevInterfaceVersion = newRefDevInterfaceVersion;
    }
    void setDutObjListId(const uint8_t newDutObjListId) { m_dutObjListId = newDutObjListId; }
    void setDutDevType(const uint8_t newDutDevType) { m_dutDevType = newDutDevType; }
    void setDutDevInterfaceVersion(const uint16_t newDutDevInterfaceVersion)
    {
        m_dutDevInterfaceVersion = newDutDevInterfaceVersion;
    }

    //========================================
    //! \brief Setter for reserved values.
    //! \attention Not part of the "public" interface For testing purpose only!
    //----------------------------------------
    void setReserved(const uint8_t r, const char newReservedValue)
    {
        if (r < nbOfReserved)
        {
            m_reserved.at(r) = newReservedValue;
        }
    }

protected:
    //! Details of the (reference) object list which contains  all reference objects of the association.
    uint8_t m_refObjListId{objListIdMask};
    uint8_t m_refDevType{0}; //!< Reference device type.

    //========================================
    //! \brief Reference device interface version.
    //! \note Only 14 bits are used.
    //----------------------------------------
    uint16_t m_refDevInterfaceVersion{devInterfaceVersionMask};

    //! Details of the object list which contains all DUT  objects of the association.
    uint8_t m_dutObjListId{objListIdMask};
    uint8_t m_dutDevType{0}; //!< DuT device type.

    //========================================
    //! \brief DuT device interface version.
    //! \note Only 14 bits are used.
    //----------------------------------------
    uint16_t m_dutDevInterfaceVersion{devInterfaceVersionMask};

    // uint32_t objAssocCount

    //! reserved bytes
    //! \note all bytes set to char(0xFF) == -1
    ReservedArray m_reserved{{static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1),
                              static_cast<char>(-1)}};

    ObjAssocVector m_objAssocs{}; //!< Vector of object associations.
}; // ObjectAssociationList4001

//==============================================================================

bool operator==(const ObjectAssociationList4001& lhs, const ObjectAssociationList4001& rhs);
bool operator!=(const ObjectAssociationList4001& lhs, const ObjectAssociationList4001& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

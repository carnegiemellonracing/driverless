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
//! \date Apr 24, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/objectlist/special/UnfilteredObjectDataIn2271.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/FilteredObjectDataIn2271.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Object for scala. With unfiltered and filtered object information.
//------------------------------------------------------------------------------
class ObjectIn2271 final
{
    friend bool operator==(const ObjectIn2271& lhs, const ObjectIn2271& rhs);

public:
    enum class InterfaceFlags : uint8_t
    {
        FilteredInterface   = 0x01U,
        UnfilteredInterface = 0x02U,
        FencesInterface     = 0x04U
    }; // InterfaceFlags

public:
    ObjectIn2271()          = default;
    virtual ~ObjectIn2271() = default;

public:
    bool hasUnfilteredAttributes() const { return m_unfilteredObjectData.isValid(); }
    bool hasUnfilteredContour() const { return m_unfilteredObjectData.hasContourPoints(); }
    bool hasFilteredAttributes() const { return m_filteredObjectData.isValid(); }
    bool hasFilteredContour() const { return m_filteredObjectData.hasContourPoints(); }

public:
    virtual std::streamsize getSerializedSize() const;
    virtual bool deserialize(std::istream& is);
    virtual bool serialize(std::ostream& os) const;

public:
    uint32_t getObjectId() const { return m_objectId; }

    uint8_t getInterfaceFlags() const { return m_interfaceFlags; }
    bool isInterfaceFlagSet(const InterfaceFlags flag) const
    {
        return (m_interfaceFlags & static_cast<uint8_t>(flag)) != 0;
    }
    bool hasFilteredInterface() const { return isInterfaceFlagSet(InterfaceFlags::FilteredInterface); }
    bool hasUnfilteredInterface() const { return isInterfaceFlagSet(InterfaceFlags::UnfilteredInterface); }
    bool hasFencesInterface() const { return isInterfaceFlagSet(InterfaceFlags::FencesInterface); }

    const UnfilteredObjectDataIn2271& getUnfilteredObjectData() const { return m_unfilteredObjectData; }
    UnfilteredObjectDataIn2271& getUnfilteredObjectData() { return m_unfilteredObjectData; }
    const FilteredObjectDataIn2271& getFilteredObjectData() const { return m_filteredObjectData; }
    FilteredObjectDataIn2271& getFilteredObjectData() { return m_filteredObjectData; }

public:
    void setObjectId(const uint32_t objectId) { m_objectId = objectId; }

    void setInterfaceFlags(const uint8_t interfaceFlags) { m_interfaceFlags = interfaceFlags; }
    void setInterfaceFlag(const InterfaceFlags flag) { m_interfaceFlags |= static_cast<uint8_t>(flag); }
    void clearInterfaceFlag(const InterfaceFlags flag)
    {
        m_interfaceFlags = static_cast<uint8_t>(m_interfaceFlags & (~static_cast<uint32_t>(flag)));
    }
    void setInterfaceFlag(const InterfaceFlags flag, const bool value)
    {
        value ? setInterfaceFlag(flag) : clearInterfaceFlag(flag);
    }
    void setHasFilteredInterface(const bool hasFilteredInterface = true)
    {
        setInterfaceFlag(InterfaceFlags::FilteredInterface, hasFilteredInterface);
    }
    void setHasUnfilteredInterface(const bool hasUnfilteredInterface = true)
    {
        setInterfaceFlag(InterfaceFlags::UnfilteredInterface, hasUnfilteredInterface);
    }
    void setHasFencesInterface(const bool hasFencesInterface = true)
    {
        setInterfaceFlag(InterfaceFlags::FencesInterface, hasFencesInterface);
    }

    void setUnfilteredObjectData(const UnfilteredObjectDataIn2271& unfilteredData)
    {
        m_unfilteredObjectData = unfilteredData;
    }
    void setFilteredObjectData(const FilteredObjectDataIn2271& filteredData) { m_filteredObjectData = filteredData; }

public: // do not use, for test purpose only
    void setInternal(const uint8_t internal) { m_internal = internal; } // do not use, for test purpose only
    uint8_t getInternal() const { return m_internal; } // do not use, for test purpose only
    void setReserved(const uint32_t reserved) { m_reserved = reserved; } // do not use, for test purpose only
    uint32_t getReserved() const { return m_reserved; } // do not use, for test purpose only

protected:
    enum class AttributeFlags : uint8_t
    {
        NoAttributes                  = 0x00U,
        UnfilteredContourAvailable    = 0x01U,
        UnfilteredAttributesAvailable = 0x02U,
        FilteredContourAvailable      = 0x04U,
        FilteredAttributesAvailable   = 0x08U,
        Undefined                     = 0x0FU
    };

protected:
    uint8_t getAttributeFlags() const;
    void setAttributeFlags(const uint8_t attrFlags);

protected:
    uint32_t m_objectId{0}; //!< Id of this object from tracking.
    uint8_t m_internal{0}; //!< Reserved for internal data.
    uint8_t m_interfaceFlags{0}; //!< The interface flags.
    //uint8_t m_attributeFlags{0};
    UnfilteredObjectDataIn2271 m_unfilteredObjectData{}; //!< The unfiltered object data.
    FilteredObjectDataIn2271 m_filteredObjectData{}; //!< The filtered object data.
    uint32_t m_reserved{0}; //!< Reserved bytes.
}; // ObjectIn2271

//==============================================================================

bool operator==(const ObjectIn2271& lhs, const ObjectIn2271& rhs);
bool operator!=(const ObjectIn2271& lhs, const ObjectIn2271& rhs);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
